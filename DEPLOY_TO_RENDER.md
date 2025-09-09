# Deploy to Render - Complete Setup Guide

This guide provides step-by-step instructions for deploying the Small-Cap Quant System to Render.

## Prerequisites

- Render account (free tier available)
- GitHub repository access
- PostgreSQL database (managed)
- Redis instance (managed) 
- API keys for data providers (Alpaca, Polygon, Tiingo)

## Architecture Overview

The system deploys as multiple Render services:

1. **Web Service**: Streamlit dashboard for monitoring and manual operations
2. **Worker Service**: Background task processing with Celery
3. **Redis Service**: Message broker for task queue
4. **PostgreSQL**: Primary database for market data and features
5. **Cron Jobs**: Automated data ingestion and trading pipeline

## Step 1: Prepare Your Repository

### 1.1 Fork/Clone Repository
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Quant.git
cd Quant
```

### 1.2 Review Configuration Files
The repository includes pre-configured Render deployment files:
- `render.yaml`: Service definitions
- `Dockerfile`: Container configuration  
- `scripts/entrypoint.sh`: Service startup logic
- `.streamlit/config.toml`: Streamlit settings

## Step 2: Set Up External Services

### 2.1 PostgreSQL Database
1. Create a PostgreSQL database on Render or external provider
2. Note the connection string: `postgresql://user:pass@host:port/dbname`

### 2.2 Redis Instance
1. Create a Redis instance on Render or external provider
2. Note the connection string: `redis://host:port/0`

### 2.3 API Keys
Gather API keys for:
- **Alpaca Markets**: Trading and market data
- **Polygon.io**: Market data and fundamentals  
- **Tiingo**: Alternative market data source

## Step 3: Deploy to Render

### 3.1 Connect Repository
1. Log into Render dashboard
2. Click "New" → "Blueprint"
3. Connect your GitHub repository
4. Select the `render.yaml` file

### 3.2 Configure Environment Variables

#### Required Variables for All Services
```env
# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Redis (Task Queue)
REDIS_URL=redis://host:6379/0
CELERY_BROKER_URL=redis://host:6379/0
CELERY_RESULT_BACKEND=redis://host:6379/0

# Alpaca Trading API
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret
APCA_API_BASE_URL=https://api.alpaca.markets
ALPACA_DATA_FEED=sip

# Market Data APIs
POLYGON_API_KEY=your_polygon_key
TIINGO_API_KEY=your_tiingo_key

# Application Settings
PYTHONPATH=/app
```

#### Optional Configuration Variables
```env
# Portfolio Settings
ALLOW_SHORTS=false
TOP_N=25
GROSS_LEVERAGE=1.0
MAX_POSITION_WEIGHT=0.10
MIN_PRICE=1.00

# Risk Management
RISK_BUDGET=1000000
MIN_ADV_USD=25000
SLIPPAGE_BPS=5

# Backtest Settings
BACKTEST_START=2019-01-01
TARGET_HORIZON_DAYS=5

# Universe Filtering
MARKET_CAP_MAX=3000000000
ADV_USD_MIN=25000
ADV_LOOKBACK=20
```

### 3.3 Service Configuration

The `render.yaml` defines these services:

#### Web Service (Streamlit Dashboard)
- **Name**: `smallcap-quant`
- **Type**: Web Service
- **Build**: Docker
- **Port**: Automatically assigned by Render
- **Environment**: `SERVICE=web APP_MODE=streamlit`

#### Data Ingestion API Service
- **Name**: `data-ingestion-service`
- **Type**: Web Service
- **Build**: Docker
- **Port**: Automatically assigned by Render
- **Environment**: `SERVICE=web APP_MODE=api`
- **Endpoints**: `/health`, `/ingest`, `/metrics`, `/status`

#### Worker Service (Background Tasks)
- **Name**: `quant-worker`  
- **Type**: Worker
- **Build**: Docker
- **Environment**: `SERVICE=worker WORKER_TASK=idle`

#### Cron Jobs
1. **Universe Refresh**: Monday 9:00 AM CT
2. **Data Ingestion**: Weekdays 4:45 PM CT  
3. **EOD Pipeline**: Weekdays 5:30 PM CT

## Step 4: Deployment Process

### 4.1 Initial Deployment
1. Deploy the blueprint in Render
2. Wait for all services to build and start
3. Monitor logs for any startup issues

### 4.2 Database Initialization
The system automatically runs Alembic migrations on startup:
```bash
# This happens automatically in the entrypoint
alembic upgrade head
```

### 4.3 Verify Services

#### Check Web Service
- Access the Streamlit dashboard via Render's provided URL
- Verify database connection and basic functionality

#### Check Data Ingestion API Service
- Test the health endpoint: `https://data-ingestion-service-se1j.onrender.com/health`
- Verify API endpoints are accessible: `https://data-ingestion-service-se1j.onrender.com/`
- Test data ingestion: `curl -X POST https://data-ingestion-service-se1j.onrender.com/ingest -H "Content-Type: application/json" -d '{"days": 1}'`

#### Check Worker Service  
- Review worker logs for successful startup
- Test task queue functionality from the dashboard

#### Verify Cron Jobs
- Confirm cron schedules in Render dashboard
- Check execution logs for any failures

## Step 5: Production Configuration

### 5.1 Security
- Use environment variables for all secrets
- Enable Render's environment variable syncing
- Regularly rotate API keys

### 5.2 Monitoring
- Set up Render monitoring alerts
- Monitor resource usage (CPU, memory)
- Track application logs for errors

### 5.3 Scaling
- **Web Service**: Auto-scales based on traffic
- **Worker Service**: Manually scale based on task load
- **Database**: Consider connection pooling for high load

### 5.4 Backup Strategy
- Enable automated database backups
- Export critical configuration data regularly
- Maintain version control of all configuration changes

## Step 6: Operations

### 6.1 Application Modes
The system supports multiple operation modes:

#### Web Dashboard (Default)
```env
SERVICE=web
APP_MODE=streamlit  # Dashboard view
```

#### Operator App
```env  
SERVICE=web
APP_MODE=operator   # Full trading interface
```

#### API Mode
```env
SERVICE=web
APP_MODE=api        # REST API only
```

### 6.2 Task Queue Operations
Enable async mode in the dashboard for:
- Market data ingestion
- Feature building
- Model training
- Trade generation

### 6.3 Manual Operations
Use the Render shell or dashboard for:
- One-time data backfills
- Model retraining
- Portfolio rebalancing

## Troubleshooting

### Common Issues

#### Build Failures
- Check Dockerfile syntax
- Verify all requirements are installable
- Review build logs for dependency conflicts

#### Database Connection Issues
- Verify DATABASE_URL format
- Check network connectivity
- Confirm database exists and user has permissions

#### API Rate Limiting
- Monitor API usage against provider limits
- Implement backoff strategies for data ingestion
- Consider upgrading API plans if needed

#### Memory Issues
- Monitor worker memory usage
- Adjust CELERY_WORKER_CONCURRENCY if needed
- Consider upgrading Render plan for more memory

#### Streamlit JavaScript Module Loading Issues
- **Symptom**: `TypeError: Failed to fetch dynamically imported module`
- **Cause**: Incorrect production configuration
- **Solution**: Ensure `.streamlit/config.toml` has production settings (see RENDER_TROUBLESHOOTING.md)

### Log Locations
- **Web Service**: Render web service logs
- **Worker Service**: Render worker logs  
- **Cron Jobs**: Individual cron job logs
- **Application**: Stdout/stderr in all services

### Support Resources
- Render Documentation: https://render.com/docs
- System Issues: Check GitHub repository issues
- Trading Questions: Consult financial documentation

## Security Considerations

1. **API Keys**: Store securely in Render environment variables
2. **Database**: Use strong passwords and connection encryption
3. **Network**: Render provides HTTPS by default
4. **Data**: Sensitive trading data should be encrypted at rest
5. **Access**: Limit dashboard access to authorized users only

## Cost Optimization

1. **Starter Tier**: Use free tier for development/testing
2. **Production**: Standard plans for production workloads
3. **Database**: Size based on data retention needs
4. **Workers**: Scale based on actual task volume
5. **Monitoring**: Use Render's built-in monitoring to track costs

## Next Steps

After successful deployment:

1. **Paper Trading**: Test with paper trading accounts first
2. **Data Validation**: Verify all data feeds are working correctly
3. **Model Performance**: Monitor prediction accuracy
4. **Risk Management**: Confirm position sizing and risk controls
5. **Production Trading**: Gradually scale to live trading

## Support

For deployment issues:
- Check the troubleshooting section above
- Review Render documentation
- Open an issue in the GitHub repository
- Contact the system maintainers

---

⚠️ **Risk Warning**: This system is for educational and research purposes. Always paper trade first and thoroughly test before using real money. Trading involves substantial risk of loss.