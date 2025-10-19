# 🚀 Vercel Deployment Checklist

## ✅ **Pre-Deployment Steps**

### 1. **Environment Variables Setup**
In Vercel Dashboard → Project Settings → Environment Variables, add:
```bash
OPENAI_API_KEY=your_actual_openai_key
PERPLEXITY_API_KEY=your_actual_perplexity_key  
POLYGON_API_KEY=your_actual_polygon_key
```

### 2. **Test Local Functionality**
```bash
# Test earnings analysis
curl -X POST "http://localhost:8000/query/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "Apple earnings Q4 2024"}'

# Test technical analysis  
curl -X POST "http://localhost:8000/query/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "RSI for AAPL"}'

# Test health endpoint
curl http://localhost:8000/health
```

### 3. **Verify File Structure**
```
requiem/
├── api/
│   ├── main.py              # Your original API
│   ├── vercel_main.py       # Vercel wrapper
│   └── routers/
│       └── earnings.py      # Earnings endpoints
├── ui/
│   ├── index.html           # Frontend
│   ├── script.js            # Frontend JS
│   └── styles.css           # Frontend CSS
├── workers/
│   └── engine/              # All your engines
├── vercel.json              # Vercel config
├── requirements.txt         # Python dependencies
└── DEPLOYMENT.md            # This guide
```

## 🚀 **Deployment Steps**

### 1. **Deploy to Vercel**
```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy
vercel

# Follow prompts:
# - Link to existing project? No
# - Project name: requiem-financial-analysis
# - Directory: ./
# - Override settings? No
```

### 2. **Configure Environment Variables**
- Go to Vercel Dashboard → Your Project → Settings → Environment Variables
- Add all three API keys from your local .env file

### 3. **Test Deployment**
```bash
# Replace YOUR_PROJECT_URL with your actual Vercel URL
YOUR_PROJECT_URL="https://your-project.vercel.app"

# Test health
curl $YOUR_PROJECT_URL/health

# Test earnings
curl -X POST "$YOUR_PROJECT_URL/query/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "Apple earnings Q4 2024"}'

# Test technical analysis
curl -X POST "$YOUR_PROJECT_URL/query/intelligent" \
  -H "Content-Type: application/json" \
  -d '{"query": "RSI for AAPL"}'
```

## ⚠️ **Critical Differences from Local**

### **What Will Work Exactly the Same:**
- ✅ All API endpoints (`/query/intelligent`, `/health`, etc.)
- ✅ Earnings analysis functionality
- ✅ Technical analysis (if TA-Lib works)
- ✅ Frontend UI and styling
- ✅ OpenAI API calls
- ✅ Perplexity API calls

### **What Will Be Different:**
- ⚠️ **ChromaDB**: Uses in-memory storage (data lost on restart)
- ⚠️ **File Storage**: No persistent file system
- ⚠️ **TA-Lib**: Uses fallback implementations (may be slightly less accurate)
- ⚠️ **Cold Starts**: First request may be slow (2-5 seconds)

### **Workarounds:**
1. **For ChromaDB**: Data will be lost between requests (serverless limitation)
2. **For TA-Lib**: ✅ **SOLVED** - Uses fallback implementations that work without C++ dependencies
3. **For Performance**: Consider upgrading to Vercel Pro for better performance

## 🔧 **Troubleshooting**

### **Common Issues:**
1. **Import Errors**: Check Python path in vercel.json
2. **Timeout Errors**: Increase function timeout in Vercel settings
3. **Memory Errors**: Increase memory allocation
4. **CORS Errors**: Check allow_origins in CORS middleware

### **Debug Commands:**
```bash
# Check deployment logs
vercel logs

# Check function logs
vercel logs --function=api/vercel_main.py

# Redeploy
vercel --prod
```

## 📊 **Expected Performance**

### **Local Development:**
- Response time: ~100-500ms
- Memory usage: ~100-200MB
- Persistent storage: ✅

### **Vercel Production:**
- Response time: ~1-3 seconds (cold start)
- Memory usage: ~50-100MB (limited)
- Persistent storage: ❌ (serverless)

## 🎯 **Success Criteria**

Your deployment is successful when:
- ✅ Health endpoint returns `{"ok": true}`
- ✅ Earnings analysis works: "Apple earnings Q4 2024"
- ✅ Technical analysis works: "RSI for AAPL"
- ✅ Frontend loads at root URL
- ✅ All API endpoints respond correctly

## 🚀 **Next Steps After Deployment**

1. **Update Frontend API URL** (if needed)
2. **Set up monitoring** and alerts
3. **Test all functionality** thoroughly
4. **Consider database migration** for persistent storage
5. **Optimize performance** based on usage

Your app will be live and accessible worldwide! 🌍✨
