# 🚀 Vercel Deployment Guide

## Prerequisites
- Vercel account (free tier available)
- GitHub repository with your code
- Environment variables ready

## Step 1: Prepare Your Repository

1. **Push your code to GitHub** (if not already done)
2. **Ensure all files are committed**:
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

## Step 2: Deploy to Vercel

### Option A: Vercel CLI (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy from your project directory
vercel

# Follow the prompts:
# - Link to existing project? No
# - Project name: requiem-financial-analysis
# - Directory: ./
# - Override settings? No
```

### Option B: Vercel Dashboard
1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repository
4. Vercel will auto-detect the configuration

## Step 3: Configure Environment Variables

In Vercel Dashboard → Project Settings → Environment Variables:

```bash
OPENAI_API_KEY=your_openai_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here
POLYGON_API_KEY=your_polygon_key_here
```

## Step 4: Update API Endpoints

Your app will be available at:
- **Frontend**: `https://your-project.vercel.app`
- **API**: `https://your-project.vercel.app/api`

Update the frontend to use the new API URL:
```javascript
// In ui/script.js, update the API base URL
const API_BASE_URL = 'https://your-project.vercel.app/api';
```

## Step 5: Test Deployment

1. **Visit your Vercel URL**
2. **Test earnings analysis**: "Apple earnings Q4 2024"
3. **Test technical analysis**: "RSI for AAPL"
4. **Check console for errors**

## 🔧 Configuration Files

### vercel.json
```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/main.py",
      "use": "@vercel/python"
    },
    {
      "src": "ui/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/main.py"
    },
    {
      "src": "/(.*)",
      "dest": "/ui/$1"
    }
  ]
}
```

### requirements.txt
```
fastapi==0.104.1
uvicorn==0.24.0
python-dotenv==1.0.0
pandas==2.1.3
numpy==1.24.3
requests==2.31.0
openai==1.3.7
chromadb==0.4.18
pdfplumber==0.10.3
beautifulsoup4==4.12.2
python-multipart==0.0.6
pydantic==2.5.0
python-dateutil==2.8.2
talib==0.4.28
```

## ⚠️ Limitations & Considerations

### What Works:
- ✅ Frontend UI (HTML, CSS, JS)
- ✅ FastAPI backend as serverless functions
- ✅ OpenAI API calls
- ✅ Perplexity API calls
- ✅ Basic earnings analysis

### What's Limited:
- ⚠️ **ChromaDB**: Uses in-memory storage (data lost on restart)
- ⚠️ **File System**: No persistent file storage
- ⚠️ **TA-Lib**: May not work on Vercel (C++ dependencies)
- ⚠️ **Cold Starts**: First request may be slow

### Workarounds:
1. **For ChromaDB**: Use external database (Pinecone, Weaviate)
2. **For TA-Lib**: Use alternative libraries or external APIs
3. **For File Storage**: Use cloud storage (AWS S3, Cloudinary)

## 🚀 Production Optimizations

### 1. Database Migration
```python
# Replace ChromaDB with Pinecone
import pinecone

pinecone.init(api_key="your-pinecone-key")
index = pinecone.Index("earnings-documents")
```

### 2. Caching
```python
# Add Redis caching
import redis
redis_client = redis.Redis(host="your-redis-url")
```

### 3. CDN
- Vercel automatically provides CDN
- Static assets are cached globally

## 📊 Monitoring

Vercel provides:
- **Analytics**: Request counts, response times
- **Functions**: Serverless function logs
- **Performance**: Core Web Vitals

## 🔄 Updates

To update your deployment:
```bash
git push origin main
# Vercel automatically redeploys
```

## 🆘 Troubleshooting

### Common Issues:
1. **Import Errors**: Check Python path in vercel.json
2. **Environment Variables**: Ensure they're set in Vercel dashboard
3. **Timeout**: Increase function timeout in Vercel settings
4. **Memory**: Increase memory allocation for complex operations

### Debug Commands:
```bash
# Check deployment logs
vercel logs

# Check function logs
vercel logs --function=api/main.py
```

## 🎯 Next Steps

1. **Deploy to Vercel** using the steps above
2. **Test all functionality** thoroughly
3. **Set up monitoring** and alerts
4. **Optimize performance** based on usage
5. **Scale up** as needed (Pro plan for higher limits)

Your financial analysis app will be live and accessible worldwide! 🌍✨
