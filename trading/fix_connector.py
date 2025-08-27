from __future__ import annotations
import socket
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from config import (
    ENABLE_FIX_PROTOCOL, FIX_HOST, FIX_PORT,
    FIX_SENDER_COMP_ID, FIX_TARGET_COMP_ID
)

log = logging.getLogger(__name__)


class OrderStatus(Enum):
    """FIX order statuses"""
    NEW = "0"
    PARTIALLY_FILLED = "1"
    FILLED = "2"
    DONE_FOR_DAY = "3"
    CANCELED = "4"
    REPLACED = "8"
    PENDING_CANCEL = "6"
    REJECTED = "8"


class OrderType(Enum):
    """FIX order types"""
    MARKET = "1"
    LIMIT = "2"
    STOP = "3"
    STOP_LIMIT = "4"


class Side(Enum):
    """FIX sides"""
    BUY = "1"
    SELL = "2"


class TimeInForce(Enum):
    """FIX time in force"""
    DAY = "0"
    GOOD_TILL_CANCEL = "1"
    IMMEDIATE_OR_CANCEL = "3"
    FILL_OR_KILL = "4"


@dataclass
class FIXMessage:
    """FIX message representation"""
    msg_type: str
    fields: Dict[int, str]
    raw_message: str = ""

    def __post_init__(self):
        if not self.raw_message:
            self.raw_message = self.to_fix_string()

    def to_fix_string(self) -> str:
        """Convert to FIX string format"""
        # Build FIX message string
        msg_parts = []

        # Add message type
        msg_parts.append(f"35={self.msg_type}")

        # Add all fields in tag order
        for tag in sorted(self.fields.keys()):
            msg_parts.append(f"{tag}={self.fields[tag]}")

        # Join with SOH (ASCII 1)
        fix_msg = "\x01".join(msg_parts) + "\x01"

        # Calculate and prepend length and checksum
        header = f"8=FIX.4.2\x019={len(fix_msg)}\x01"

        # Calculate checksum
        checksum = sum(ord(c) for c in header + fix_msg) % 256

        return f"{header}{fix_msg}10={checksum:03d}\x01"


@dataclass
class OrderRequest:
    """Order request for FIX execution"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    order_type: str = "market"  # 'market', 'limit'
    price: Optional[float] = None
    client_order_id: Optional[str] = None
    time_in_force: str = "day"


class FIXConnector:
    """FIX protocol connector for low-latency order execution"""

    def __init__(self,
                 host: str = None,
                 port: int = None,
                 sender_comp_id: str = None,
                 target_comp_id: str = None):

        self.host = host or FIX_HOST
        self.port = port or FIX_PORT
        self.sender_comp_id = sender_comp_id or FIX_SENDER_COMP_ID
        self.target_comp_id = target_comp_id or FIX_TARGET_COMP_ID

        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.sequence_number = 1
        self.session_active = False

        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}
        self.order_callbacks: Dict[str, Callable] = {}

        # Threading
        self.receiver_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.running = False

        # Message buffer
        self.message_buffer = ""

        # Setup default handlers
        self._setup_default_handlers()

    def _setup_default_handlers(self):
        """Setup default message handlers"""
        self.message_handlers["A"] = self._handle_logon
        self.message_handlers["0"] = self._handle_heartbeat
        self.message_handlers["1"] = self._handle_test_request
        self.message_handlers["8"] = self._handle_execution_report
        self.message_handlers["9"] = self._handle_order_cancel_reject

    def connect(self) -> bool:
        """Connect to FIX server"""
        if not ENABLE_FIX_PROTOCOL:
            log.info("FIX protocol disabled in config")
            return False

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(30.0)
            self.socket.connect((self.host, self.port))

            self.connected = True
            self.running = True

            # Start receiver thread
            self.receiver_thread = threading.Thread(target=self._receive_messages)
            self.receiver_thread.daemon = True
            self.receiver_thread.start()

            # Start heartbeat thread
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
            self.heartbeat_thread.daemon = True
            self.heartbeat_thread.start()

            # Send logon message
            self._send_logon()

            log.info(f"FIX connection established to {self.host}:{self.port}")
            return True

        except Exception as e:
            log.error(f"Failed to connect to FIX server: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from FIX server"""
        if self.connected:
            try:
                self._send_logout()
                time.sleep(1)  # Give time for logout to be processed
            except Exception:
                pass

        self.running = False
        self.connected = False
        self.session_active = False

        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass

        log.info("FIX connection closed")

    def send_order(self, order: OrderRequest, callback: Optional[Callable] = None) -> str:
        """
        Send order via FIX protocol

        Args:
            order: Order request
            callback: Optional callback for order updates

        Returns:
            Client order ID
        """
        if not self.session_active:
            raise Exception("FIX session not active")

        # Generate client order ID if not provided
        client_order_id = order.client_order_id or f"ORDER_{int(time.time() * 1000)}"

        # Register callback
        if callback:
            self.order_callbacks[client_order_id] = callback

        # Build new order single message (D)
        fields = {
            11: client_order_id,  # ClOrdID
            55: order.symbol,     # Symbol
            54: Side.BUY.value if order.side.lower() == 'buy' else Side.SELL.value,  # Side
            38: str(order.quantity),  # OrderQty
            40: OrderType.MARKET.value if order.order_type.lower() == 'market' else OrderType.LIMIT.value,  # OrdType
            59: TimeInForce.DAY.value,  # TimeInForce
            60: datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3],  # TransactTime
        }

        # Add price for limit orders
        if order.order_type.lower() == 'limit' and order.price:
            fields[44] = f"{order.price:.2f}"  # Price

        # Add sender/target comp IDs
        fields[49] = self.sender_comp_id  # SenderCompID
        fields[56] = self.target_comp_id  # TargetCompID
        fields[34] = str(self.sequence_number)  # MsgSeqNum
        fields[52] = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3]  # SendingTime

        message = FIXMessage("D", fields)
        self._send_message(message)

        log.info(f"Sent order: {client_order_id} for {order.quantity} {order.symbol} {order.side}")
        return client_order_id

    def cancel_order(self, client_order_id: str, symbol: str) -> str:
        """Cancel an existing order"""
        if not self.session_active:
            raise Exception("FIX session not active")

        cancel_id = f"CANCEL_{int(time.time() * 1000)}"

        fields = {
            11: cancel_id,           # ClOrdID
            41: client_order_id,     # OrigClOrdID
            55: symbol,              # Symbol
            54: Side.BUY.value,      # Side (required but ignored for cancel)
            60: datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3],  # TransactTime
            49: self.sender_comp_id,
            56: self.target_comp_id,
            34: str(self.sequence_number),
            52: datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3],
        }

        message = FIXMessage("F", fields)  # Order Cancel Request
        self._send_message(message)

        return cancel_id

    def _send_message(self, message: FIXMessage):
        """Send FIX message"""
        if not self.connected:
            raise Exception("Not connected to FIX server")

        try:
            self.socket.send(message.raw_message.encode('ascii'))
            self.sequence_number += 1
            log.debug(f"Sent FIX message: {message.msg_type}")
        except Exception as e:
            log.error(f"Failed to send FIX message: {e}")
            raise

    def _send_logon(self):
        """Send logon message"""
        fields = {
            98: "0",  # EncryptMethod (None)
            108: "30",  # HeartBtInt
            49: self.sender_comp_id,
            56: self.target_comp_id,
            34: str(self.sequence_number),
            52: datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3],
        }

        message = FIXMessage("A", fields)  # Logon
        self._send_message(message)

    def _send_logout(self):
        """Send logout message"""
        fields = {
            49: self.sender_comp_id,
            56: self.target_comp_id,
            34: str(self.sequence_number),
            52: datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3],
        }

        message = FIXMessage("5", fields)  # Logout
        self._send_message(message)

    def _send_heartbeat(self, test_req_id: str = None):
        """Send heartbeat message"""
        fields = {
            49: self.sender_comp_id,
            56: self.target_comp_id,
            34: str(self.sequence_number),
            52: datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3],
        }

        if test_req_id:
            fields[112] = test_req_id  # TestReqID

        message = FIXMessage("0", fields)  # Heartbeat
        self._send_message(message)

    def _receive_messages(self):
        """Receive and process FIX messages"""
        while self.running and self.connected:
            try:
                data = self.socket.recv(4096)
                if not data:
                    break

                self.message_buffer += data.decode('ascii')
                self._process_buffer()

            except socket.timeout:
                continue
            except Exception as e:
                log.error(f"Error receiving FIX messages: {e}")
                break

        self.connected = False

    def _process_buffer(self):
        """Process received message buffer"""
        while '\x01' in self.message_buffer:
            # Find complete message
            soh_pos = self.message_buffer.find('\x01')
            if soh_pos == -1:
                break

            # Extract potential message
            potential_msg = self.message_buffer[:soh_pos + 1]

            # Simple message validation (should be more robust)
            if potential_msg.startswith('8=FIX'):
                try:
                    message = self._parse_fix_message(potential_msg)
                    self._handle_message(message)
                    self.message_buffer = self.message_buffer[soh_pos + 1:]
                except Exception as e:
                    log.error(f"Failed to parse FIX message: {e}")
                    self.message_buffer = self.message_buffer[1:]
            else:
                self.message_buffer = self.message_buffer[1:]

    def _parse_fix_message(self, raw_msg: str) -> FIXMessage:
        """Parse raw FIX message"""
        fields = {}
        msg_type = ""

        parts = raw_msg.split('\x01')
        for part in parts:
            if '=' in part:
                tag, value = part.split('=', 1)
                tag = int(tag)
                fields[tag] = value

                if tag == 35:  # MsgType
                    msg_type = value

        return FIXMessage(msg_type, fields, raw_msg)

    def _handle_message(self, message: FIXMessage):
        """Handle received FIX message"""
        handler = self.message_handlers.get(message.msg_type)
        if handler:
            try:
                handler(message)
            except Exception as e:
                log.error(f"Error handling FIX message {message.msg_type}: {e}")
        else:
            log.debug(f"No handler for FIX message type: {message.msg_type}")

    def _handle_logon(self, message: FIXMessage):
        """Handle logon response"""
        self.session_active = True
        log.info("FIX session established")

    def _handle_heartbeat(self, message: FIXMessage):
        """Handle heartbeat message"""
        log.debug("Received heartbeat")

    def _handle_test_request(self, message: FIXMessage):
        """Handle test request"""
        test_req_id = message.fields.get(112)
        self._send_heartbeat(test_req_id)

    def _handle_execution_report(self, message: FIXMessage):
        """Handle execution report"""
        client_order_id = message.fields.get(11)
        status = message.fields.get(39)

        log.info(f"Execution report for {client_order_id}: status {status}")

        # Call order callback if registered
        if client_order_id and client_order_id in self.order_callbacks:
            try:
                self.order_callbacks[client_order_id](message)
            except Exception as e:
                log.error(f"Error in order callback: {e}")

    def _handle_order_cancel_reject(self, message: FIXMessage):
        """Handle order cancel reject"""
        client_order_id = message.fields.get(11)
        reject_reason = message.fields.get(102, "Unknown")

        log.warning(f"Order cancel reject for {client_order_id}: {reject_reason}")

    def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running and self.connected:
            if self.session_active:
                try:
                    self._send_heartbeat()
                except Exception as e:
                    log.error(f"Failed to send heartbeat: {e}")

            time.sleep(30)  # 30 second heartbeat interval


# Global FIX connector instance
_fix_connector: Optional[FIXConnector] = None


def get_fix_connector() -> Optional[FIXConnector]:
    """Get global FIX connector instance"""
    global _fix_connector

    if not ENABLE_FIX_PROTOCOL:
        return None

    if _fix_connector is None:
        _fix_connector = FIXConnector()

    return _fix_connector


def init_fix_connection() -> bool:
    """Initialize FIX connection"""
    connector = get_fix_connector()
    if connector and not connector.connected:
        return connector.connect()
    return connector is not None and connector.connected


def close_fix_connection():
    """Close FIX connection"""
    global _fix_connector
    if _fix_connector:
        _fix_connector.disconnect()
        _fix_connector = None
