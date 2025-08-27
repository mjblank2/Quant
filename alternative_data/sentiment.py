"""
NLP Sentiment Analysis for Alternative Alpha Generation
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, date
import logging
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

@dataclass
class SentimentScore:
    """Sentiment score with metadata"""
    score: float  # -1 to 1 scale
    confidence: float  # 0 to 1 confidence
    source: str
    timestamp: datetime
    keywords: List[str]
    entity_mentions: List[str]

class SentimentAnalyzer:
    """Base sentiment analyzer - uses SimpleSentimentAnalyzer by default"""
    
    def __init__(self):
        self._analyzer = SimpleSentimentAnalyzer()
    
    def analyze_text(self, text: str, symbols: Optional[List[str]] = None) -> Dict[str, SentimentScore]:
        """Analyze sentiment of text for given symbols"""
        return self._analyzer.analyze_text(text, symbols)

class SimpleSentimentAnalyzer(SentimentAnalyzer):
    """
    Simple rule-based sentiment analyzer
    In production, would use more sophisticated NLP models
    """
    
    def __init__(self):
        # Positive/negative word lists (simplified)
        self.positive_words = {
            'strong', 'growth', 'profit', 'beat', 'outperform', 'bullish', 'buy',
            'upgrade', 'exceed', 'momentum', 'robust', 'solid', 'impressive',
            'breakthrough', 'innovative', 'expansion', 'surge', 'rally'
        }
        
        self.negative_words = {
            'weak', 'decline', 'loss', 'miss', 'underperform', 'bearish', 'sell',
            'downgrade', 'below', 'pressure', 'concern', 'risk', 'drop',
            'crash', 'collapse', 'struggle', 'warning', 'threat', 'slump'
        }
        
        # Intensity modifiers
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.3, 'significantly': 1.4,
            'substantially': 1.6, 'dramatically': 1.8, 'slightly': 0.7
        }
        
        # Negation words
        self.negations = {'not', 'no', 'never', 'none', 'nothing', 'neither', 'nor'}
    
    def analyze_text(self, text: str, symbols: Optional[List[str]] = None) -> Dict[str, SentimentScore]:
        """Analyze sentiment with simple rule-based approach"""
        if not text.strip():
            return {}
        
        # Clean and tokenize text
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Find symbol mentions
        mentioned_symbols = []
        if symbols:
            for symbol in symbols:
                if symbol.lower() in text_lower:
                    mentioned_symbols.append(symbol)
        
        if not mentioned_symbols:
            # If no specific symbols mentioned, analyze general sentiment
            mentioned_symbols = ['MARKET']
        
        results = {}
        
        for symbol in mentioned_symbols:
            sentiment_score = self._calculate_sentiment_score(words, text_lower)
            keywords = self._extract_keywords(words)
            
            results[symbol] = SentimentScore(
                score=sentiment_score,
                confidence=min(0.6, len(keywords) * 0.1 + 0.2),  # Simple confidence
                source='rule_based',
                timestamp=datetime.now(),
                keywords=keywords,
                entity_mentions=[symbol] if symbol != 'MARKET' else []
            )
        
        return results
    
    def _calculate_sentiment_score(self, words: List[str], text: str) -> float:
        """Calculate sentiment score from words"""
        score = 0.0
        i = 0
        
        while i < len(words):
            word = words[i]
            
            # Check for intensifiers
            intensity = 1.0
            if i > 0 and words[i-1] in self.intensifiers:
                intensity = self.intensifiers[words[i-1]]
            
            # Check for negations (look back 2-3 words)
            negated = False
            for j in range(max(0, i-3), i):
                if words[j] in self.negations:
                    negated = True
                    break
            
            # Score the word
            if word in self.positive_words:
                word_score = 1.0 * intensity
                if negated:
                    word_score = -word_score
                score += word_score
            elif word in self.negative_words:
                word_score = -1.0 * intensity
                if negated:
                    word_score = -word_score
                score += word_score
            
            i += 1
        
        # Normalize score to [-1, 1] range
        return np.tanh(score / max(1, len(words) * 0.1))
    
    def _extract_keywords(self, words: List[str]) -> List[str]:
        """Extract sentiment-relevant keywords"""
        keywords = []
        for word in words:
            if word in self.positive_words or word in self.negative_words:
                keywords.append(word)
        return keywords[:10]  # Limit to top 10

class NewsProcessor:
    """Process news articles for sentiment analysis"""
    
    def __init__(self, sentiment_analyzer: Optional[SentimentAnalyzer] = None):
        self.sentiment_analyzer = sentiment_analyzer or SimpleSentimentAnalyzer()
        
    def process_news_feed(self, news_articles: List[Dict[str, Any]], 
                         symbol_universe: List[str]) -> pd.DataFrame:
        """
        Process a feed of news articles
        
        Args:
            news_articles: List of articles with 'title', 'content', 'timestamp', 'source'
            symbol_universe: List of symbols to analyze sentiment for
            
        Returns:
            DataFrame with sentiment scores by symbol and date
        """
        sentiment_records = []
        
        for article in news_articles:
            title = article.get('title', '')
            content = article.get('content', '')
            timestamp = pd.to_datetime(article.get('timestamp', datetime.now()))
            source = article.get('source', 'unknown')
            
            # Combine title and content for analysis
            full_text = f"{title}. {content}"
            
            # Analyze sentiment
            sentiment_results = self.sentiment_analyzer.analyze_text(full_text, symbol_universe)
            
            for symbol, sentiment in sentiment_results.items():
                sentiment_records.append({
                    'symbol': symbol,
                    'date': timestamp.date(),
                    'timestamp': timestamp,
                    'sentiment_score': sentiment.score,
                    'confidence': sentiment.confidence,
                    'source': source,
                    'keywords': ','.join(sentiment.keywords[:5]),
                    'article_title': title[:100]  # Truncate for storage
                })
        
        if not sentiment_records:
            return pd.DataFrame()
        
        df = pd.DataFrame(sentiment_records)
        
        # Aggregate multiple articles per symbol/date
        daily_sentiment = df.groupby(['symbol', 'date']).agg({
            'sentiment_score': 'mean',
            'confidence': 'mean',
            'timestamp': 'count'  # Number of articles
        }).reset_index()
        
        daily_sentiment.rename(columns={'timestamp': 'article_count'}, inplace=True)
        
        return daily_sentiment
    
    def calculate_sentiment_features(self, sentiment_df: pd.DataFrame, 
                                   lookback_days: List[int] = [1, 3, 7, 14]) -> pd.DataFrame:
        """Calculate rolling sentiment features"""
        if sentiment_df.empty:
            return pd.DataFrame()
        
        # Ensure proper date column
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df = sentiment_df.sort_values(['symbol', 'date'])
        
        features = []
        
        for symbol, group in sentiment_df.groupby('symbol'):
            group = group.set_index('date').sort_index()
            
            feature_row = {'symbol': symbol}
            
            for days in lookback_days:
                # Rolling sentiment metrics
                rolling_sentiment = group['sentiment_score'].rolling(f'{days}D').mean()
                rolling_volatility = group['sentiment_score'].rolling(f'{days}D').std()
                rolling_trend = group['sentiment_score'].diff(days)
                
                feature_row[f'sentiment_mean_{days}d'] = rolling_sentiment.iloc[-1] if not rolling_sentiment.empty else np.nan
                feature_row[f'sentiment_vol_{days}d'] = rolling_volatility.iloc[-1] if not rolling_volatility.empty else np.nan
                feature_row[f'sentiment_trend_{days}d'] = rolling_trend.iloc[-1] if not rolling_trend.empty else np.nan
            
            # Latest values
            if not group.empty:
                feature_row['latest_sentiment'] = group['sentiment_score'].iloc[-1]
                feature_row['latest_confidence'] = group['confidence'].iloc[-1]
                feature_row['latest_article_count'] = group['article_count'].iloc[-1]
                feature_row['latest_date'] = group.index[-1]
            
            features.append(feature_row)
        
        return pd.DataFrame(features)

# Example news sentiment integration
class MockNewsAPI:
    """Mock news API for demonstration"""
    
    def get_recent_news(self, symbols: List[str], days: int = 7) -> List[Dict[str, Any]]:
        """Get recent news for symbols (mock implementation)"""
        # In practice, this would integrate with real news APIs
        # like Bloomberg, Reuters, Alpha Architect, etc.
        
        sample_news = []
        
        for symbol in symbols[:5]:  # Limit for demo
            # Generate mock articles
            sample_news.extend([
                {
                    'title': f'{symbol} Reports Strong Q3 Earnings Beat',
                    'content': f'{symbol} exceeded analyst expectations with strong revenue growth and improved margins.',
                    'timestamp': datetime.now(),
                    'source': 'financial_times'
                },
                {
                    'title': f'Analyst Upgrades {symbol} on Innovation Pipeline',
                    'content': f'Investment bank raises price target for {symbol} citing breakthrough products.',
                    'timestamp': datetime.now(),
                    'source': 'reuters'
                }
            ])
        
        return sample_news

def integrate_sentiment_with_features(feature_store, symbols: List[str], 
                                    sentiment_analyzer: Optional[SentimentAnalyzer] = None) -> pd.DataFrame:
    """
    Integration function to add sentiment features to existing feature store
    """
    if sentiment_analyzer is None:
        sentiment_analyzer = SimpleSentimentAnalyzer()
    
    # Get recent news (mock)
    news_api = MockNewsAPI()
    recent_news = news_api.get_recent_news(symbols)
    
    # Process sentiment
    news_processor = NewsProcessor(sentiment_analyzer)
    sentiment_df = news_processor.process_news_feed(recent_news, symbols)
    
    # Calculate features
    sentiment_features = news_processor.calculate_sentiment_features(sentiment_df)
    
    return sentiment_features