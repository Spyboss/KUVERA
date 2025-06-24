# 🚀 Railway Deployment Guide for Kuvera Grid Trading Bot

This guide will help you deploy your Kuvera Grid Trading Bot to Railway with an integrated web dashboard.

## 🌟 What You Get

✅ **24/7 Cloud Hosting** - No need to keep your PC running  
✅ **Web Dashboard** - Monitor your bot from anywhere  
✅ **Real-time Logs** - See what your bot is doing  
✅ **Trade History** - Track all your trades  
✅ **Bot Controls** - Start/stop/restart from the web interface  
✅ **Mobile Friendly** - Access from your phone  
✅ **Free Tier Available** - $5/month credit from Railway  

## 📋 Prerequisites

1. **GitHub Account** - To store your code
2. **Railway Account** - Sign up at [railway.app](https://railway.app)
3. **Binance API Keys** - From your Binance account
4. **OpenRouter API Key** - For AI features (optional)

## 🚀 Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Initialize Git** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Create GitHub Repository**:
   - Go to GitHub and create a new repository
   - Push your code:
   ```bash
   git remote add origin https://github.com/yourusername/kuvera-grid-bot.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy to Railway

1. **Sign up for Railway**:
   - Go to [railway.app](https://railway.app)
   - Sign up with your GitHub account
   - You get $5/month free credit!

2. **Create New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your Kuvera Grid repository
   - Railway will automatically detect the Dockerfile

3. **Configure Environment Variables**:
   Click on your project → Variables tab and add:
   
   **Required Variables:**
   ```
   BINANCE_API_KEY=your_binance_api_key
   BINANCE_SECRET_KEY=your_binance_secret_key
   BINANCE_TESTNET_API_KEY=your_testnet_api_key
   BINANCE_TESTNET_SECRET_KEY=your_testnet_secret_key
   SECRET_KEY=your_flask_secret_key_here
   ```
   
   **Optional Variables:**
   ```
   AUTOMATED_MODE=true
   TRADING_MODE=testnet
   AI_ENABLED=true
   STRATEGY_TYPE=mean_reversion
   START_BOT=true
   FLASK_DEBUG=false
   OPENROUTER_API_KEY=your_openrouter_key
   NEWS_API_KEY=your_news_api_key
   ```

4. **Deploy**:
   - Railway will automatically build and deploy
   - Wait for the build to complete (5-10 minutes)
   - Your bot will be available at the provided Railway URL

3. **Configure Environment Variables**:
   Click on your project → Variables tab → Add these:
   ```
   BINANCE_API_KEY=your_binance_api_key_here
   BINANCE_SECRET_KEY=your_binance_secret_key_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   FLASK_ENV=production
   START_BOT=true
   PORT=5000
   ```

4. **Deploy**:
   - Railway will automatically build and deploy your app
   - Wait for the build to complete (usually 2-3 minutes)

### Step 3: Access Your Dashboard

1. **Get Your URL**:
   - In Railway dashboard, click on your project
   - Go to "Deployments" tab
   - Click on the latest deployment
   - Copy the public URL (looks like: `https://your-app-name.up.railway.app`)

2. **Open Dashboard**:
   - Visit your URL in a browser
   - You'll see the Kuvera Grid Trading Bot Dashboard! 🎉

## 🖥️ Dashboard Features

### Main Dashboard
- **Bot Status**: Running/Stopped with live indicator
- **Financial Metrics**: Balance, P&L, Win Rate
- **Trading Activity**: Today's trades, active positions
- **Controls**: Start/Stop/Restart buttons

### Real-time Monitoring
- **Live Logs**: See bot activity in real-time
- **Trade History**: All your trades with timestamps
- **Performance Metrics**: Updated every 30 seconds
- **Mobile Responsive**: Works on phones and tablets

### Bot Controls
- **Start Bot**: Begin trading
- **Stop Bot**: Pause trading
- **Restart Bot**: Restart if issues occur

## 📱 Mobile Access

1. **Bookmark the URL** on your phone
2. **Add to Home Screen** for app-like experience:
   - **iPhone**: Safari → Share → Add to Home Screen
   - **Android**: Chrome → Menu → Add to Home Screen

## 🔧 Configuration

### Trading Settings
Edit these in Railway's environment variables:
- `TRADING_SYMBOL=BTCUSDT`
- `TRADING_CAPITAL=100.0`
- `RISK_PER_TRADE=0.015`

### Bot Behavior
- `START_BOT=true` - Auto-start bot on deployment
- `FLASK_DEBUG=false` - Keep false for production

## 📊 Monitoring & Alerts

### Built-in Monitoring
- **Health Checks**: Railway monitors your app
- **Auto-restart**: If bot crashes, it restarts automatically
- **Logs**: All activity logged and viewable

### Custom Alerts (Optional)
Add these to your bot code:
```python
# Email alerts
import smtplib
from email.mime.text import MIMEText

def send_alert(message):
    # Configure with your email settings
    pass

# Telegram alerts
import requests

def send_telegram_alert(message):
    bot_token = "YOUR_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": message})
```

## 💰 Cost Breakdown

### Railway Pricing
- **Free Tier**: $5/month credit (enough for small bots)
- **Pro Plan**: $20/month for unlimited usage
- **Usage-based**: Only pay for what you use

### Typical Monthly Cost
- **Small Bot**: $0-5/month (within free tier)
- **Active Bot**: $5-15/month
- **High-frequency Bot**: $15-30/month

**Compare to running locally**: $20-40/month in electricity!

## 🛠️ Troubleshooting

### Common Issues

1. **Build Failed**:
   - Check requirements.txt for typos
   - Ensure all files are committed to Git
   - Check Railway build logs

2. **Bot Not Starting**:
   - Verify API keys in environment variables
   - Check logs for error messages
   - Ensure `START_BOT=true`

3. **Dashboard Not Loading**:
   - Check if deployment is running
   - Verify PORT environment variable
   - Check health endpoint: `/health`

### Getting Help

1. **Railway Logs**:
   - Go to Railway dashboard
   - Click on your project
   - View "Logs" tab for detailed information

2. **Health Check**:
   - Visit: `https://your-app.up.railway.app/health`
   - Should return JSON with status

3. **Support**:
   - Railway Discord: [discord.gg/railway](https://discord.gg/railway)
   - Railway Docs: [docs.railway.app](https://docs.railway.app)

## 🔄 Updates & Maintenance

### Updating Your Bot
1. Make changes to your code locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update bot logic"
   git push
   ```
3. Railway automatically redeploys!

### Monitoring Performance
- Check dashboard daily
- Monitor Railway usage in their dashboard
- Review logs for any errors

## 🎯 Next Steps

1. **Deploy your bot** following this guide
2. **Test with small amounts** first
3. **Monitor performance** via dashboard
4. **Scale up** once comfortable
5. **Add custom alerts** for important events

## 🔐 Security Best Practices

1. **Never commit API keys** to Git
2. **Use environment variables** for all secrets
3. **Enable 2FA** on all accounts
4. **Regular monitoring** of trades and balances
5. **Start with testnet** before live trading

---

🎉 **Congratulations!** You now have a professional, cloud-hosted trading bot with a beautiful web dashboard!

**Dashboard URL**: `https://your-app-name.up.railway.app`  
**Health Check**: `https://your-app-name.up.railway.app/health`  
**API Status**: `https://your-app-name.up.railway.app/api/status`

Happy Trading! 🚀📈