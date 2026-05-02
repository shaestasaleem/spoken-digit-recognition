# DEPLOYMENT CONFIGURATION GUIDE
# PhonemeIQ v3.0 - Professional Spoken Digit Recognition
# ═══════════════════════════════════════════════════════════════════

## 1. STREAMLIT CLOUD (RECOMMENDED - FREE & EASIEST)

### Prerequisites:
- GitHub account
- Repository with all files

### Steps:
1. Push code to GitHub:
   ```bash
   git add .
   git commit -m "Initial PhonemeIQ commit"
   git push origin main
   ```

2. Go to share.streamlit.io
3. Click "New App"
4. Select GitHub repo
5. Set main file: app_professional.py
6. Deploy!

### Expected Startup Time: 2-3 minutes
### Free Tier: Yes
### Custom Domain: No (but can use ngrok)


## 2. DOCKER & DOCKER COMPOSE (LOCAL/CLOUD)

### Local Development:
```bash
# Build image
docker build -t phonemeiq:latest .

# Run container
docker run -p 8501:8501 phonemeiq:latest

# Or use docker-compose
docker-compose up
```

### Push to Docker Hub:
```bash
docker tag phonemeiq:latest USERNAME/phonemeiq:latest
docker push USERNAME/phonemeiq:latest
```

### Expected Startup Time: 1-2 minutes
### Resource Requirements: 2GB RAM, 1 CPU minimum


## 3. HEROKU DEPLOYMENT

### Prerequisites:
- Heroku CLI installed
- Procfile (already provided)
- requirements.txt (already updated)

### Steps:
```bash
heroku login
heroku create phonemeiq-app
git push heroku main
```

### Expected Startup Time: 3-5 minutes
### Free Tier: Sleeping dynos (older plan)
### Hobby Plan: $7/month


## 4. GOOGLE CLOUD RUN

### Prerequisites:
- Google Cloud SDK
- Google Cloud project
- Billing enabled

### Steps:
```bash
gcloud init  # If first time
gcloud run deploy phonemeiq \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 3600
```

### Expected Startup Time: 5-10 minutes
### Cost: Pay-per-use (~$0.00001 per request)
### File: app.yaml (already provided)


## 5. AWS ELASTIC BEANSTALK

### Prerequisites:
- AWS account
- AWS CLI installed
- EB CLI installed

### Steps:
```bash
eb init -p python-3.10 phonemeiq
eb create phonemeiq-env
eb deploy
```

### Expected Startup Time: 5-10 minutes
### Free Tier: Available
### Cost after free tier: ~$10-30/month


## 6. AZURE CONTAINER INSTANCES

### Prerequisites:
- Azure account
- Azure CLI

### Steps:
```bash
az group create --name phonemeiq-rg --location eastus
az container create \
  --resource-group phonemeiq-rg \
  --name phonemeiq \
  --image YOUR_REGISTRY/phonemeiq:latest \
  --ports 8501 \
  --environment-variables \
    STREAMLIT_SERVER_MAXUPLOADSIZE=100
```

### Expected Startup Time: 3-5 minutes
### Cost: ~$0.0015/hour (estimated)


## 7. RENDER.COM (NEW HEROKU ALTERNATIVE)

### Prerequisites:
- Render account
- GitHub repository

### Steps:
1. Go to render.com
2. Create new Web Service
3. Connect GitHub repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `streamlit run app_professional.py`
6. Set env vars in Dashboard

### Expected Startup Time: 2-3 minutes
### Free Tier: Yes (auto-sleeps after 15 min inactivity)


## COMPARISON TABLE

| Platform | Cost | Setup Time | Performance | Scaling | Custom Domain |
|----------|------|-----------|-------------|---------|---------------|
| Streamlit Cloud | Free | 5 min | Good | Automatic | Yes |
| Docker Hub + Cloud | $0-50/mo | 15 min | Excellent | Manual | Yes |
| Heroku | Free-$7/mo | 10 min | Fair | Limited | Yes |
| Google Cloud Run | Pay-per-use | 15 min | Excellent | Automatic | Yes |
| AWS EB | Free-30/mo | 20 min | Good | Good | Yes |
| Azure | ~$40/mo | 20 min | Excellent | Good | Yes |
| Render | Free-$12/mo | 10 min | Good | Good | Yes |


## ENVIRONMENT VARIABLES (Optional)

Create .env file or set in deployment platform:

```env
STREAMLIT_SERVER_MAXUPLOADSIZE=100
STREAMLIT_SERVER_ENABLECORS=false
STREAMLIT_LOGGER_LEVEL=warning
STREAMLIT_CLIENT_SHOWSTDOUT=false
STREAMLIT_CLIENT_SHOWERRDETAILS=false
```


## PERFORMANCE OPTIMIZATION FOR DEPLOYMENT

1. **Reduce Model Size:**
   ```python
   # Use quantization for TensorFlow model
   converter = tf.lite.TFLiteConverter.from_saved_model("model_dir")
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   ```

2. **Cache Models:**
   - Already implemented with @st.cache_resource
   - Loads models once on startup

3. **Enable Caching:**
   ```toml
   [client]
   toolbarMode = "minimal"
   showMenuItems = false
   ```

4. **Optimize Dependencies:**
   - Consider using lighter alternatives
   - Remove unused packages


## MONITORING & HEALTH CHECKS

### Streamlit Cloud:
- Automatic monitoring
- View logs in dashboard

### Docker/Cloud Run:
```dockerfile
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
```

### Heroku:
```json
"formation": {
  "web": {
    "process_type": "web",
    "quantity": 1,
    "size": "standard-1x"
  }
}
```


## SCALING RECOMMENDATIONS

**Low Traffic (< 1000 users/month):**
- Use Streamlit Cloud or Render Free
- Auto-scaling sufficient

**Medium Traffic (1000-10000 users/month):**
- Docker on single Cloud Run instance
- ~2GB RAM, 1 CPU

**High Traffic (> 10000 users/month):**
- Kubernetes cluster
- Multiple replicas
- Load balancer
- Consider API wrapper instead of Streamlit


## TROUBLESHOOTING DEPLOYMENTS

### Common Issues:

1. **Import errors:**
   - Ensure all packages in requirements.txt
   - Check Python version compatibility

2. **Model loading fails:**
   - Verify model files in deployment
   - Check file permissions
   - Ensure correct paths (use relative paths)

3. **Memory issues:**
   - Increase allocated RAM
   - Check model sizes
   - Monitor during startup

4. **Slow startup:**
   - Cache models (@st.cache_resource)
   - Use lightweight frameworks
   - Optimize imports

5. **Port binding issues:**
   - Ensure using dynamic port (not hardcoded)
   - Use $PORT environment variable


## GETTING SUPPORT

1. Check Streamlit docs: docs.streamlit.io
2. Check platform-specific docs
3. Review application logs
4. Check Docker logs: `docker logs container_id`
5. Try local setup first before deploying


## SECURITY CHECKLIST

- [ ] Remove debug mode (STREAMLIT_LOGGER_LEVEL=warning)
- [ ] Enable CSRF protection
- [ ] Set maxUploadSize
- [ ] Use environment variables for secrets
- [ ] Enable HTTPS/TLS
- [ ] Monitor error logs
- [ ] Rate limiting enabled
- [ ] Regular security updates


---

**Last Updated:** January 2025
**PhonemeIQ v3.0** | Harvard-LUMS Speech Processing Laboratory
