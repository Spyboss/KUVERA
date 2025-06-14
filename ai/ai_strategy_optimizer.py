"""AI Strategy Optimizer - OpenRouter Integration
Provides AI-enhanced trading decisions using multiple models
"""

import asyncio
import aiohttp
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

class AIStrategyOptimizer:
    """AI-powered strategy optimization using OpenRouter API"""
    
    def __init__(self, api_key: str, rate_limit_delay: float = 12.0):
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay  # 12 seconds for free tier
        self.last_request_time = 0
        self.logger = logging.getLogger('AIOptimizer')
        
        # AI model configurations
        self.models = {
            'sentiment': 'google/gemma-7b-it',
            'strategy': 'mistralai/mistral-7b-instruct:free', 
            'anomaly': 'huggingfaceh4/zephyr-7b-beta'
        }
        
        # Cache for AI responses
        self.response_cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # AI confidence tracking
        self.ai_confidence_history = []
        self.market_sentiment_score = 0.5  # Neutral
        
    async def _make_request(self, model: str, prompt: str, max_tokens: int = 150) -> Optional[str]:
        """Make rate-limited request to OpenRouter API"""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - time_since_last)
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://kuvera-grid.local',
                'X-Title': 'Kuvera Grid Trading Bot'
            }
            
            payload = {
                'model': model,
                'messages': [
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': max_tokens,
                'temperature': 0.3  # Lower temperature for more consistent responses
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.last_request_time = time.time()
                        return data['choices'][0]['message']['content'].strip()
                    else:
                        self.logger.error(f"API request failed: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error making AI request: {e}")
            return None
    
    async def analyze_market_sentiment(self, price_data: List[float], news_summary: str = "") -> float:
        """Analyze market sentiment using AI"""
        try:
            # Check cache first
            cache_key = f"sentiment_{len(price_data)}_{hash(news_summary)}"
            if self._is_cached(cache_key):
                return self.response_cache[cache_key]['data']
            
            # Calculate basic metrics
            if len(price_data) < 5:
                return 0.5  # Neutral if insufficient data
                
            recent_change = (price_data[-1] - price_data[-5]) / price_data[-5] * 100
            volatility = np.std(price_data[-10:]) / np.mean(price_data[-10:]) * 100
            
            prompt = f"""
Analyze Bitcoin market sentiment based on:
- Recent 5-period price change: {recent_change:.2f}%
- Current volatility: {volatility:.2f}%
- News context: {news_summary[:200] if news_summary else 'No recent news'}

Provide sentiment score (0.0=very bearish, 0.5=neutral, 1.0=very bullish) and brief reasoning.
Format: SCORE:0.XX REASON:your_analysis
"""
            
            response = await self._make_request(self.models['sentiment'], prompt)
            if response:
                # Parse response
                try:
                    if 'SCORE:' in response:
                        score_str = response.split('SCORE:')[1].split()[0]
                        sentiment_score = float(score_str)
                        sentiment_score = max(0.0, min(1.0, sentiment_score))  # Clamp to [0,1]
                        
                        # Cache result
                        self._cache_response(cache_key, sentiment_score)
                        self.market_sentiment_score = sentiment_score
                        
                        self.logger.info(f"AI Sentiment Analysis: {sentiment_score:.2f} - {response}")
                        return sentiment_score
                except ValueError:
                    pass
            
            # Fallback to technical analysis
            return self._calculate_technical_sentiment(price_data)
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return 0.5  # Neutral on error
    
    async def optimize_strategy_parameters(self, current_params: Dict, performance_data: Dict) -> Dict:
        """Optimize strategy parameters using AI"""
        try:
            cache_key = f"strategy_{hash(str(current_params))}_{hash(str(performance_data))}"
            if self._is_cached(cache_key):
                return self.response_cache[cache_key]['data']
            
            win_rate = performance_data.get('win_rate', 0)
            profit = performance_data.get('total_profit', 0)
            trades = performance_data.get('trade_count', 0)
            
            prompt = f"""
Optimize trading strategy parameters:
Current: SMA={current_params.get('sma_period', 20)}, Entry={current_params.get('entry_threshold', 0.005)*100:.1f}%, Exit={current_params.get('exit_threshold', 0.005)*100:.1f}%, StopLoss={current_params.get('stop_loss', 0.008)*100:.1f}%
Performance: WinRate={win_rate:.1f}%, Profit=${profit:.2f}, Trades={trades}

Suggest optimized parameters for better performance.
Format: SMA:XX ENTRY:X.X EXIT:X.X STOP:X.X CONFIDENCE:XX
"""
            
            response = await self._make_request(self.models['strategy'], prompt)
            if response and 'SMA:' in response:
                try:
                    # Parse AI suggestions
                    optimized = current_params.copy()
                    
                    if 'SMA:' in response:
                        sma = int(response.split('SMA:')[1].split()[0])
                        optimized['sma_period'] = max(5, min(50, sma))
                    
                    if 'ENTRY:' in response:
                        entry = float(response.split('ENTRY:')[1].split()[0]) / 100
                        optimized['entry_threshold'] = max(0.001, min(0.02, entry))
                    
                    if 'EXIT:' in response:
                        exit_val = float(response.split('EXIT:')[1].split()[0]) / 100
                        optimized['exit_threshold'] = max(0.001, min(0.02, exit_val))
                    
                    if 'STOP:' in response:
                        stop = float(response.split('STOP:')[1].split()[0]) / 100
                        optimized['stop_loss'] = max(0.005, min(0.05, stop))
                    
                    # Cache and return
                    self._cache_response(cache_key, optimized)
                    self.logger.info(f"AI Strategy Optimization: {response}")
                    return optimized
                    
                except (ValueError, IndexError) as e:
                    self.logger.error(f"Error parsing AI strategy response: {e}")
            
            return current_params  # Return unchanged on error
            
        except Exception as e:
            self.logger.error(f"Error in strategy optimization: {e}")
            return current_params
    
    async def detect_market_anomaly(self, price_data: List[float], volume_data: List[float] = None) -> Tuple[bool, float, str]:
        """Detect market anomalies that might require emergency stops"""
        try:
            if len(price_data) < 10:
                return False, 0.0, "Insufficient data"
            
            # Calculate anomaly indicators
            recent_change = (price_data[-1] - price_data[-5]) / price_data[-5] * 100
            volatility = np.std(price_data[-10:]) / np.mean(price_data[-10:]) * 100
            
            # Quick local checks for obvious anomalies
            if abs(recent_change) > 10:  # >10% change in 5 periods
                return True, 0.9, f"Extreme price movement: {recent_change:.1f}%"
            
            if volatility > 5:  # Very high volatility
                return True, 0.8, f"High volatility detected: {volatility:.1f}%"
            
            # AI analysis for subtle anomalies
            cache_key = f"anomaly_{len(price_data)}_{recent_change:.2f}_{volatility:.2f}"
            if self._is_cached(cache_key):
                cached = self.response_cache[cache_key]['data']
                return cached['anomaly'], cached['confidence'], cached['reason']
            
            prompt = f"""
Detect market anomalies in Bitcoin price data:
- Recent 5-period change: {recent_change:.2f}%
- Current volatility: {volatility:.2f}%
- Price trend: {price_data[-5:]}

Is this anomalous behavior requiring emergency action?
Format: ANOMALY:YES/NO CONFIDENCE:XX REASON:explanation
"""
            
            response = await self._make_request(self.models['anomaly'], prompt, max_tokens=100)
            if response:
                try:
                    is_anomaly = 'ANOMALY:YES' in response.upper()
                    confidence = 0.5
                    reason = "AI analysis"
                    
                    if 'CONFIDENCE:' in response:
                        conf_str = response.split('CONFIDENCE:')[1].split()[0]
                        confidence = float(conf_str) / 100
                    
                    if 'REASON:' in response:
                        reason = response.split('REASON:')[1].strip()
                    
                    result = {
                        'anomaly': is_anomaly,
                        'confidence': confidence,
                        'reason': reason
                    }
                    
                    self._cache_response(cache_key, result)
                    self.logger.info(f"AI Anomaly Detection: {is_anomaly} ({confidence:.2f}) - {reason}")
                    
                    return is_anomaly, confidence, reason
                    
                except (ValueError, IndexError):
                    pass
            
            return False, 0.0, "No anomaly detected"
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            return False, 0.0, f"Error: {e}"
    
    def _calculate_technical_sentiment(self, price_data: List[float]) -> float:
        """Fallback technical sentiment calculation"""
        try:
            if len(price_data) < 10:
                return 0.5
            
            # Simple momentum-based sentiment
            short_ma = np.mean(price_data[-5:])
            long_ma = np.mean(price_data[-10:])
            
            if short_ma > long_ma:
                return 0.6  # Slightly bullish
            else:
                return 0.4  # Slightly bearish
                
        except Exception:
            return 0.5
    
    def _is_cached(self, key: str) -> bool:
        """Check if response is cached and still valid"""
        if key not in self.response_cache:
            return False
        
        cache_time = self.response_cache[key]['timestamp']
        return (time.time() - cache_time) < self.cache_duration
    
    def _cache_response(self, key: str, data):
        """Cache AI response with timestamp"""
        self.response_cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
        
        # Clean old cache entries
        current_time = time.time()
        expired_keys = [
            k for k, v in self.response_cache.items()
            if current_time - v['timestamp'] > self.cache_duration
        ]
        for k in expired_keys:
            del self.response_cache[k]
    
    def get_ai_confidence_score(self) -> float:
        """Get overall AI confidence score"""
        try:
            # Combine sentiment confidence with recent performance
            base_confidence = 0.7  # Base confidence
            sentiment_factor = abs(self.market_sentiment_score - 0.5) * 0.4  # 0-0.2 boost
            
            return min(0.95, base_confidence + sentiment_factor)
            
        except Exception:
            return 0.7  # Default confidence
    
    async def validate_trade_signal(self, signal_type: str, price: float, sma: float, 
                                  confidence_threshold: float = 0.7) -> Tuple[bool, float, str]:
        """Validate trade signal using AI analysis"""
        try:
            # Quick technical validation
            price_vs_sma = (price - sma) / sma * 100
            
            prompt = f"""
Validate {signal_type.upper()} signal for Bitcoin:
- Current price: ${price:.2f}
- SMA-20: ${sma:.2f}
- Price vs SMA: {price_vs_sma:.2f}%
- Market sentiment: {self.market_sentiment_score:.2f}

Should we execute this {signal_type} signal?
Format: EXECUTE:YES/NO CONFIDENCE:XX REASON:explanation
"""
            
            response = await self._make_request(self.models['strategy'], prompt, max_tokens=100)
            if response:
                try:
                    should_execute = 'EXECUTE:YES' in response.upper()
                    confidence = 0.5
                    reason = "AI validation"
                    
                    if 'CONFIDENCE:' in response:
                        conf_str = response.split('CONFIDENCE:')[1].split()[0]
                        confidence = float(conf_str) / 100
                    
                    if 'REASON:' in response:
                        reason = response.split('REASON:')[1].strip()
                    
                    # Apply confidence threshold
                    final_decision = should_execute and confidence >= confidence_threshold
                    
                    self.logger.info(f"AI Signal Validation: {final_decision} ({confidence:.2f}) - {reason}")
                    return final_decision, confidence, reason
                    
                except (ValueError, IndexError):
                    pass
            
            # Fallback to basic validation
            basic_confidence = 0.6 if abs(price_vs_sma) > 0.3 else 0.4
            return basic_confidence >= confidence_threshold, basic_confidence, "Technical validation"
            
        except Exception as e:
            self.logger.error(f"Error in signal validation: {e}")
            return False, 0.0, f"Validation error: {e}"