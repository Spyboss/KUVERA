"""Autonomous Trading Engine - 24/7 Automated Trading with AI Integration
Handles automatic signal detection, validation, and execution
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import deque

from .ai_strategy_optimizer import AIStrategyOptimizer
from .emergency_stop import EmergencyStopSystem

class AutonomousTrader:
    """Fully autonomous trading engine with AI integration"""
    
    def __init__(self, bot_instance, config: Dict, openrouter_api_key: str = None):
        self.bot = bot_instance
        self.config = config
        self.logger = logging.getLogger('AutoTrader')
        
        # AI Integration
        self.ai_optimizer = None
        if openrouter_api_key:
            self.ai_optimizer = AIStrategyOptimizer(openrouter_api_key)
            self.logger.info("AI Strategy Optimizer initialized")
        else:
            self.logger.warning("No OpenRouter API key - running without AI enhancement")
        
        # Emergency Stop System
        emergency_config = {
            'max_daily_loss_pct': config.get('autonomous', {}).get('max_daily_loss', 0.03),
            'max_consecutive_losses': config.get('autonomous', {}).get('max_consecutive_losses', 5),
            'max_drawdown_pct': config.get('autonomous', {}).get('max_drawdown', 0.05),
            'volatility_threshold': config.get('autonomous', {}).get('volatility_threshold', 5.0),
            'flash_crash_threshold': config.get('autonomous', {}).get('flash_crash_threshold', 0.10)
        }
        self.emergency_system = EmergencyStopSystem(emergency_config)
        
        # Autonomous trading state
        self.is_autonomous = False
        self.auto_trading_enabled = True
        self.signal_validation_enabled = True
        self.ai_confidence_threshold = config.get('autonomous', {}).get('ai_confidence_threshold', 0.7)
        
        # Signal detection parameters
        self.signal_strength_threshold = config.get('autonomous', {}).get('signal_strength_threshold', 0.6)
        self.min_signal_gap = config.get('autonomous', {}).get('min_signal_gap_seconds', 300)  # 5 minutes
        self.last_signal_time = 0
        
        # Performance tracking
        self.autonomous_stats = {
            'signals_detected': 0,
            'signals_executed': 0,
            'ai_vetoed_signals': 0,
            'emergency_stops': 0,
            'total_autonomous_profit': 0.0,
            'start_time': datetime.now()
        }
        
        # Market data buffers for AI analysis
        self.price_buffer_extended = deque(maxlen=200)  # Extended buffer for AI
        self.volume_buffer = deque(maxlen=100)
        self.signal_history = deque(maxlen=50)
        
        # Strategy optimization
        self.last_optimization_time = 0
        self.optimization_interval = 86400  # 24 hours
        self.current_strategy_params = config.get('strategy', {}).copy()
        
    async def start_autonomous_trading(self):
        """Start autonomous trading mode"""
        self.is_autonomous = True
        self.autonomous_stats['start_time'] = datetime.now()
        
        self.logger.info("🤖 AUTONOMOUS TRADING MODE ACTIVATED")
        self.logger.info(f"AI Enhancement: {'ENABLED' if self.ai_optimizer else 'DISABLED'}")
        self.logger.info(f"Signal Validation: {'ENABLED' if self.signal_validation_enabled else 'DISABLED'}")
        self.logger.info(f"AI Confidence Threshold: {self.ai_confidence_threshold:.1%}")
        
        # Start autonomous trading loop
        asyncio.create_task(self._autonomous_trading_loop())
        
        # Start periodic optimization if AI is available
        if self.ai_optimizer:
            asyncio.create_task(self._strategy_optimization_loop())
    
    def stop_autonomous_trading(self):
        """Stop autonomous trading mode"""
        self.is_autonomous = False
        self.logger.info("🛑 AUTONOMOUS TRADING MODE DEACTIVATED")
        self._log_autonomous_summary()
    
    async def _autonomous_trading_loop(self):
        """Main autonomous trading loop"""
        self.logger.info("Starting autonomous trading loop")
        
        while self.is_autonomous and self.bot.is_running:
            try:
                # Check emergency conditions first
                emergency_triggered, emergency_reasons = self.emergency_system.check_emergency_conditions()
                if emergency_triggered:
                    self.logger.critical(f"EMERGENCY STOP: {'; '.join(emergency_reasons)}")
                    self.autonomous_stats['emergency_stops'] += 1
                    await self._handle_emergency_stop(emergency_reasons)
                    continue
                
                # Update emergency system with current data
                if self.bot.current_price:
                    self.emergency_system.update_price_data(self.bot.current_price)
                    self.emergency_system.update_balance(self._get_current_balance())
                
                # Process market data and signals
                await self._process_autonomous_signals()
                
                # Sleep for a short interval
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in autonomous trading loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _process_autonomous_signals(self):
        """Process trading signals autonomously"""
        try:
            if not self.auto_trading_enabled or self.bot.paused:
                return
            
            current_price = self.bot.current_price
            if not current_price:
                return
            
            # Update extended price buffer
            self.price_buffer_extended.append(current_price)
            
            # Calculate SMA
            sma = self.bot.calculate_sma(self.current_strategy_params.get('sma_period', 20))
            if not sma:
                return
            
            # Detect signals
            buy_signal = await self._detect_buy_signal(current_price, sma)
            sell_signal = await self._detect_sell_signal(current_price, sma)
            
            # Process buy signal
            if buy_signal:
                await self._process_buy_signal(current_price, sma)
            
            # Process sell signal
            if sell_signal:
                await self._process_sell_signal(current_price, sma)
                
        except Exception as e:
            self.logger.error(f"Error processing autonomous signals: {e}")
    
    async def _detect_buy_signal(self, current_price: float, sma: float) -> bool:
        """Detect buy signal with enhanced logic"""
        try:
            # Basic conditions
            if self.bot.position or not self._check_signal_timing():
                return False
            
            # Enhanced signal detection
            entry_threshold = self.current_strategy_params.get('entry_threshold', 0.005)
            buy_threshold = sma * (1 - entry_threshold)
            
            # Basic signal
            basic_signal = current_price <= buy_threshold
            if not basic_signal:
                return False
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength('BUY', current_price, sma)
            if signal_strength < self.signal_strength_threshold:
                self.logger.debug(f"Buy signal too weak: {signal_strength:.2f}")
                return False
            
            self.autonomous_stats['signals_detected'] += 1
            self.logger.info(f"BUY SIGNAL DETECTED - Strength: {signal_strength:.2f}, Price: ${current_price:.2f}, SMA: ${sma:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error detecting buy signal: {e}")
            return False
    
    async def _detect_sell_signal(self, current_price: float, sma: float) -> bool:
        """Detect sell signal with enhanced logic"""
        try:
            if not self.bot.position or not self._check_signal_timing():
                return False
            
            # Check both take profit and stop loss
            should_sell, reason = self.bot.should_sell(current_price, sma)
            if not should_sell:
                return False
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength('SELL', current_price, sma)
            if signal_strength < self.signal_strength_threshold:
                self.logger.debug(f"Sell signal too weak: {signal_strength:.2f}")
                return False
            
            self.autonomous_stats['signals_detected'] += 1
            self.logger.info(f"SELL SIGNAL DETECTED ({reason}) - Strength: {signal_strength:.2f}, Price: ${current_price:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error detecting sell signal: {e}")
            return False
    
    def _calculate_signal_strength(self, signal_type: str, current_price: float, sma: float) -> float:
        """Calculate signal strength based on multiple factors"""
        try:
            strength = 0.0
            
            # Price deviation from SMA
            price_deviation = abs(current_price - sma) / sma
            strength += min(0.4, price_deviation * 20)  # Max 0.4 points
            
            # Volume confirmation (if available)
            if len(self.volume_buffer) > 5:
                recent_volume = np.mean(list(self.volume_buffer)[-5:])
                avg_volume = np.mean(list(self.volume_buffer))
                if recent_volume > avg_volume * 1.2:  # 20% above average
                    strength += 0.2
            
            # Trend consistency
            if len(self.price_buffer_extended) >= 10:
                prices = list(self.price_buffer_extended)[-10:]
                if signal_type == 'BUY':
                    # Look for downtrend before buy
                    trend_score = (prices[0] - prices[-1]) / prices[0]
                    strength += min(0.2, trend_score * 10)
                else:
                    # Look for uptrend before sell
                    trend_score = (prices[-1] - prices[0]) / prices[0]
                    strength += min(0.2, trend_score * 10)
            
            # Market volatility factor
            if len(self.price_buffer_extended) >= 20:
                prices = list(self.price_buffer_extended)[-20:]
                volatility = np.std(prices) / np.mean(prices)
                if volatility < 0.02:  # Low volatility = stronger signal
                    strength += 0.2
            
            return min(1.0, strength)
            
        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return 0.5  # Default moderate strength
    
    async def _process_buy_signal(self, current_price: float, sma: float):
        """Process buy signal with AI validation"""
        try:
            # AI validation if available
            if self.ai_optimizer and self.signal_validation_enabled:
                should_execute, ai_confidence, reason = await self.ai_optimizer.validate_trade_signal(
                    'BUY', current_price, sma, self.ai_confidence_threshold
                )
                
                if not should_execute:
                    self.autonomous_stats['ai_vetoed_signals'] += 1
                    self.logger.warning(f"AI VETOED BUY SIGNAL: {reason} (Confidence: {ai_confidence:.2f})")
                    return
                
                self.logger.info(f"AI APPROVED BUY SIGNAL: {reason} (Confidence: {ai_confidence:.2f})")
            
            # Check emergency system for position sizing
            should_reduce, size_multiplier = self.emergency_system.should_reduce_position_size()
            if should_reduce:
                self.logger.info(f"Risk management: Reducing position size to {size_multiplier:.1%}")
            
            # Execute buy order
            await self._execute_autonomous_buy(current_price, size_multiplier)
            
        except Exception as e:
            self.logger.error(f"Error processing buy signal: {e}")
    
    async def _process_sell_signal(self, current_price: float, sma: float):
        """Process sell signal with AI validation"""
        try:
            # AI validation if available
            if self.ai_optimizer and self.signal_validation_enabled:
                should_execute, ai_confidence, reason = await self.ai_optimizer.validate_trade_signal(
                    'SELL', current_price, sma, self.ai_confidence_threshold
                )
                
                if not should_execute:
                    self.autonomous_stats['ai_vetoed_signals'] += 1
                    self.logger.warning(f"AI VETOED SELL SIGNAL: {reason} (Confidence: {ai_confidence:.2f})")
                    return
                
                self.logger.info(f"AI APPROVED SELL SIGNAL: {reason} (Confidence: {ai_confidence:.2f})")
            
            # Execute sell order
            await self._execute_autonomous_sell(current_price)
            
        except Exception as e:
            self.logger.error(f"Error processing sell signal: {e}")
    
    async def _execute_autonomous_buy(self, current_price: float, size_multiplier: float = 1.0):
        """Execute autonomous buy order"""
        try:
            # Calculate position size with risk adjustment
            base_position_size = self.bot.calculate_position_size(current_price)
            adjusted_size = base_position_size * size_multiplier
            
            if adjusted_size <= 0:
                self.logger.warning("Position size too small after risk adjustment")
                return
            
            # Place order using bot's method but with autonomous tracking
            original_method = self.bot.place_buy_order
            
            # Override position size calculation temporarily
            self.bot.calculate_position_size = lambda price: adjusted_size
            
            await self.bot.place_buy_order(current_price)
            
            # Restore original method
            self.bot.calculate_position_size = original_method
            
            # Update autonomous stats
            self.autonomous_stats['signals_executed'] += 1
            self.last_signal_time = time.time()
            
            # Log autonomous execution
            self.logger.info(f"🤖 AUTONOMOUS BUY EXECUTED: {adjusted_size:.6f} at ${current_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error executing autonomous buy: {e}")
    
    async def _execute_autonomous_sell(self, current_price: float):
        """Execute autonomous sell order"""
        try:
            # Get current position info before selling
            position_info = self.bot.position.copy() if self.bot.position else None
            entry_price = self.bot.entry_price
            
            # Execute sell using bot's method
            await self.bot.place_sell_order("AUTONOMOUS")
            
            # Calculate and track profit
            if position_info and entry_price:
                quantity = position_info['quantity']
                profit = (current_price - entry_price) * quantity
                self.autonomous_stats['total_autonomous_profit'] += profit
                
                # Update emergency system
                self.emergency_system.add_trade_result(profit, "AUTONOMOUS")
            
            # Update autonomous stats
            self.autonomous_stats['signals_executed'] += 1
            self.last_signal_time = time.time()
            
            # Log autonomous execution
            self.logger.info(f"🤖 AUTONOMOUS SELL EXECUTED at ${current_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error executing autonomous sell: {e}")
    
    def _check_signal_timing(self) -> bool:
        """Check if enough time has passed since last signal"""
        return (time.time() - self.last_signal_time) >= self.min_signal_gap
    
    async def _handle_emergency_stop(self, reasons: List[str]):
        """Handle emergency stop situation"""
        try:
            # Stop autonomous trading temporarily
            self.auto_trading_enabled = False
            
            # Close any open positions if configured to do so
            if self.bot.position and self.config.get('autonomous', {}).get('emergency_close_positions', True):
                self.logger.critical("Emergency closing position")
                await self.bot.place_sell_order("EMERGENCY_STOP")
            
            # Send emergency alert
            await self.bot.send_alert(f"🚨 EMERGENCY STOP: {'; '.join(reasons)}")
            
            # Wait for conditions to improve
            await self._wait_for_recovery()
            
        except Exception as e:
            self.logger.error(f"Error handling emergency stop: {e}")
    
    async def _wait_for_recovery(self):
        """Wait for market conditions to improve"""
        self.logger.info("Waiting for market conditions to improve...")
        
        while self.is_autonomous:
            try:
                can_resume, reason = self.emergency_system.can_resume_trading()
                if can_resume:
                    self.logger.info(f"Market conditions improved: {reason}")
                    self.auto_trading_enabled = True
                    await self.bot.send_alert("✅ Autonomous trading resumed")
                    break
                
                self.logger.debug(f"Still waiting for recovery: {reason}")
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error waiting for recovery: {e}")
                await asyncio.sleep(60)
    
    async def _strategy_optimization_loop(self):
        """Periodic strategy optimization using AI"""
        while self.is_autonomous and self.bot.is_running:
            try:
                current_time = time.time()
                if current_time - self.last_optimization_time >= self.optimization_interval:
                    await self._optimize_strategy_parameters()
                    self.last_optimization_time = current_time
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in strategy optimization loop: {e}")
                await asyncio.sleep(3600)
    
    async def _optimize_strategy_parameters(self):
        """Optimize strategy parameters using AI"""
        try:
            if not self.ai_optimizer:
                return
            
            # Gather performance data
            performance_data = {
                'win_rate': (self.bot.winning_trades / self.bot.trade_count * 100) if self.bot.trade_count > 0 else 0,
                'total_profit': self.bot.total_profit,
                'trade_count': self.bot.trade_count
            }
            
            # Get AI optimization suggestions
            optimized_params = await self.ai_optimizer.optimize_strategy_parameters(
                self.current_strategy_params, performance_data
            )
            
            # Apply optimizations if they're reasonable
            if self._validate_parameter_changes(optimized_params):
                self.current_strategy_params.update(optimized_params)
                self.logger.info(f"Strategy parameters optimized: {optimized_params}")
            else:
                self.logger.warning("AI suggested parameters rejected - too extreme")
                
        except Exception as e:
            self.logger.error(f"Error optimizing strategy parameters: {e}")
    
    def _validate_parameter_changes(self, new_params: Dict) -> bool:
        """Validate that parameter changes are reasonable"""
        try:
            # Check SMA period
            sma_period = new_params.get('sma_period', 20)
            if not (5 <= sma_period <= 50):
                return False
            
            # Check thresholds
            entry_threshold = new_params.get('entry_threshold', 0.005)
            exit_threshold = new_params.get('exit_threshold', 0.005)
            stop_loss = new_params.get('stop_loss', 0.008)
            
            if not (0.001 <= entry_threshold <= 0.02):
                return False
            if not (0.001 <= exit_threshold <= 0.02):
                return False
            if not (0.005 <= stop_loss <= 0.05):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _get_current_balance(self) -> float:
        """Get current account balance"""
        try:
            # This would need to be implemented based on your bot's balance tracking
            # For now, return a placeholder
            return self.config.get('trading', {}).get('capital', 30.0) + self.bot.total_profit
        except Exception:
            return 30.0
    
    def _log_autonomous_summary(self):
        """Log summary of autonomous trading session"""
        try:
            duration = datetime.now() - self.autonomous_stats['start_time']
            
            self.logger.info("🤖 AUTONOMOUS TRADING SUMMARY:")
            self.logger.info(f"  Duration: {duration}")
            self.logger.info(f"  Signals Detected: {self.autonomous_stats['signals_detected']}")
            self.logger.info(f"  Signals Executed: {self.autonomous_stats['signals_executed']}")
            self.logger.info(f"  AI Vetoed Signals: {self.autonomous_stats['ai_vetoed_signals']}")
            self.logger.info(f"  Emergency Stops: {self.autonomous_stats['emergency_stops']}")
            self.logger.info(f"  Autonomous Profit: ${self.autonomous_stats['total_autonomous_profit']:.2f}")
            
            execution_rate = (self.autonomous_stats['signals_executed'] / 
                            max(1, self.autonomous_stats['signals_detected']) * 100)
            self.logger.info(f"  Execution Rate: {execution_rate:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error logging autonomous summary: {e}")
    
    def get_autonomous_status(self) -> Dict:
        """Get current autonomous trading status"""
        return {
            'is_autonomous': self.is_autonomous,
            'auto_trading_enabled': self.auto_trading_enabled,
            'ai_enabled': self.ai_optimizer is not None,
            'emergency_stopped': self.emergency_system.is_emergency_stopped,
            'stats': self.autonomous_stats.copy(),
            'risk_assessment': self.emergency_system.get_risk_assessment(),
            'current_strategy_params': self.current_strategy_params.copy()
        }