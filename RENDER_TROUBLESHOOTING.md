# Render Deployment Troubleshooting Guide

This guide helps resolve common issues when deploying the Small-Cap Quant System to Render.

## Pre-deployment Checklist

Before troubleshooting deployment issues, run the verification script:

```bash
python scripts/verify_render_setup.py
```

This will check for common configuration issues.

## Common Build Issues

### 1. Docker Build Failures

#### SSL Certificate Issues
**Symptoms**: Build fails with SSL certificate verification errors
```
ERROR: Could not find a version that satisfies the requirement numpy
SSLError: certificate verify failed
```

**Solutions**:
- This is usually an environment issue in the build system
- In Render, the build should work normally
- For local testing, try: `pip install --trusted-host pypi.org --trusted-host pypi.python.org`

#### Requirements Installation Failures  
**Symptoms**: pip install fails for specific packages
```
ERROR: No matching distribution found for package_name
```

**Solutions**:
1. Check `requirements.txt` for typos
2. Verify package versions are compatible with Python 3.12
3. Check if package exists on PyPI
4. Consider adding version constraints

#### Memory Issues During Build
**Symptoms**: Build fails with out-of-memory errors
```
Killed
Error: Process completed with exit code 137
```

**Solutions**:
1. Upgrade to a higher Render plan with more build resources
2. Optimize Dockerfile to reduce memory usage
3. Consider using pre-built base images

### 2. Application Startup Issues

#### Database Connection Failures
**Symptoms**: Application fails to start with database errors
```
sqlalchemy.exc.OperationalError: (psycopg.OperationalError) connection failed
```

**Solutions**:
1. Verify `DATABASE_URL` is correctly set
2. Check database server is running and accessible
3. Verify database credentials
4. Ensure database exists
5. Check network connectivity

#### Missing Environment Variables
**Symptoms**: Application starts but features don't work
```
KeyError: 'POLYGON_API_KEY'
AttributeError: module 'config' has no attribute 'REDIS_URL'
```

**Solutions**:
1. Check all required environment variables are set in Render
2. Verify variable names match exactly (case-sensitive)
3. Use `.env.render.template` as reference
4. Check `sync: false` is set for sensitive variables

#### Redis Connection Issues
**Symptoms**: Task queue features fail
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solutions**:
1. Verify Redis service is running
2. Check `REDIS_URL` environment variable
3. Verify Redis service name in `render.yaml`
4. Test Redis connection with health check endpoint

## Service-Specific Issues

### Web Service Issues

#### Streamlit Startup Failures
**Symptoms**: Web service fails to start
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solutions**:
1. Verify `streamlit` is in `requirements.txt`
2. Check `SERVICE=web` and `APP_MODE=streamlit` are set
3. Verify entrypoint script has correct Streamlit command

#### Dynamic Module Loading Failures
**Symptoms**: Streamlit starts but fails to load JavaScript modules
```
TypeError: Failed to fetch dynamically imported module: https://your-app.onrender.com/static/js/index.XXXXXXXX.js
```

**Solutions**:
1. Update `.streamlit/config.toml` with production settings:
   ```toml
   [server]
   enableCORS = true
   enableXsrfProtection = true  
   address = "0.0.0.0"
   headless = true
   runOnSave = false
   allowRunOnSave = false
   enableStaticServing = false
   
   [browser]
   gatherUsageStats = false
   
   [global]
   developmentMode = false
   
   [runner]
   magicEnabled = false
   fastReruns = false
   ```
2. Add `--server.headless true` flag to Streamlit commands in entrypoint
3. Ensure `--browser.gatherUsageStats false` is set for security
4. Clear browser cache and test again

#### Health Check Failures
**Symptoms**: Service shows as unhealthy in Render
```
Health check failed: GET /health returned 503
```

**Solutions**:
1. Check if database is accessible
2. Verify `health_api.py` is working
3. Test health endpoint locally
4. Check application logs for errors

#### Port Binding Issues
**Symptoms**: Service starts but is not accessible
```
Address already in use
```

**Solutions**:
1. Verify Render assigns `PORT` environment variable
2. Check entrypoint uses `${PORT}` correctly
3. Ensure application binds to `0.0.0.0:${PORT}`

### Worker Service Issues

#### Celery Worker Failures
**Symptoms**: Background tasks don't execute
```
celery.exceptions.WorkerLostError
```

**Solutions**:
1. Check `WORKER_TASK=celery` is set
2. Verify Redis connection
3. Check worker logs for errors
4. Verify task definitions are importable

#### Task Import Errors
**Symptoms**: Worker starts but can't find tasks
```
ImportError: No module named 'tasks'
```

**Solutions**:
1. Verify `PYTHONPATH=/app` is set
2. Check task modules exist and are importable
3. Verify task decorators are correctly applied

### Cron Job Issues

#### Cron Jobs Not Executing
**Symptoms**: Scheduled tasks don't run
```
Cron job failed to start
```

**Solutions**:
1. Check cron schedule syntax in `render.yaml`
2. Verify `dockerCommand` is correct
3. Check timezone settings (Render uses UTC)
4. Test commands manually in worker shell

#### Pipeline Failures
**Symptoms**: EOD pipeline fails
```
Pipeline execution failed with error
```

**Solutions**:
1. Check API rate limits aren't exceeded
2. Verify all required API keys are set
3. Check data connectivity
4. Review pipeline logs for specific errors

## API and Data Issues

### Market Data API Issues

#### Alpaca API Failures
**Symptoms**: Trading operations fail
```
alpaca_trade_api.rest.APIError: 401 Unauthorized
```

**Solutions**:
1. Verify `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`
2. Check API keys are for correct environment (paper/live)
3. Verify account status with Alpaca
4. Check API usage limits

#### Polygon API Rate Limiting
**Symptoms**: Data ingestion fails intermittently
```
requests.exceptions.HTTPError: 429 Too Many Requests
```

**Solutions**:
1. Implement backoff and retry logic
2. Upgrade Polygon plan for higher limits
3. Reduce ingestion frequency
4. Batch requests more efficiently

#### Data Quality Issues
**Symptoms**: Models produce poor results
```
Warning: Large number of NaN values in features
```

**Solutions**:
1. Check data ingestion logs
2. Verify API responses contain expected data
3. Check for market holidays/weekends
4. Review data validation settings

## Performance Issues

### Memory Usage
**Symptoms**: Application crashes with memory errors
```
MemoryError: Unable to allocate array
```

**Solutions**:
1. Upgrade to higher Render plan
2. Optimize feature engineering to use less memory
3. Process data in smaller batches
4. Add memory monitoring

### Database Performance
**Symptoms**: Slow query performance
```
Queries taking too long to execute
```

**Solutions**:
1. Add database indexes for frequently queried columns
2. Optimize query patterns
3. Consider database connection pooling
4. Upgrade database plan

### Task Queue Performance
**Symptoms**: Tasks pile up in queue
```
Tasks not being processed fast enough
```

**Solutions**:
1. Scale worker service to more instances
2. Increase worker concurrency
3. Optimize task processing time
4. Consider task prioritization

## Monitoring and Debugging

### Log Analysis

#### Application Logs
```bash
# In Render dashboard, check logs for each service:
# - Web service: Streamlit startup and request logs
# - Worker service: Celery worker and task execution logs  
# - Cron jobs: Individual job execution logs
```

#### Common Log Patterns
```
# Successful startup
[entrypoint] SERVICE=web APP_MODE=streamlit PORT=10000
[entrypoint] Running Alembic upgrade...
[entrypoint] Starting Streamlit dashboard on port 10000

# Database issues
sqlalchemy.exc.OperationalError: connection failed
psycopg.OperationalError: server closed the connection

# API issues  
alpaca_trade_api.rest.APIError: 401 Unauthorized
requests.exceptions.HTTPError: 429 Too Many Requests
```

### Health Monitoring

#### Health Check Endpoint
```bash
# Test health endpoint
curl https://your-app.onrender.com/health

# Expected response
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "service": "web",
  "checks": {
    "database": "healthy",
    "redis": "healthy"
  }
}
```

#### Status Monitoring
```bash
# Detailed status
curl https://your-app.onrender.com/status

# Includes data freshness and configuration
```

### Performance Monitoring

#### Resource Usage
- Monitor CPU and memory usage in Render dashboard
- Set up alerts for resource limits
- Track response times

#### Database Monitoring
- Monitor connection counts
- Track query performance
- Set up database alerts

## Recovery Procedures

### Database Recovery
1. **Backup**: Ensure regular database backups are enabled
2. **Migration Issues**: Use Alembic to rollback/forward migrations
3. **Data Corruption**: Restore from backup and re-run data ingestion

### Service Recovery
1. **Restart Services**: Use Render dashboard to restart failed services
2. **Rollback**: Deploy previous working version if needed
3. **Scale Resources**: Temporarily increase resources during recovery

### Data Recovery
1. **Re-ingestion**: Use dashboard to trigger data backfill
2. **Feature Rebuild**: Regenerate features from clean data
3. **Model Retraining**: Retrain models with corrected data

## Getting Help

### Internal Resources
1. Check application logs first
2. Review this troubleshooting guide
3. Use verification script to check configuration
4. Test health endpoints

### External Support
1. **Render Support**: For platform-specific issues
2. **API Provider Support**: For data feed issues
3. **Community Forums**: For general troubleshooting
4. **GitHub Issues**: For code-related problems

### Emergency Contacts
- Keep contact information for:
  - Database administrator
  - API provider support
  - System maintainer
  - Render account owner

## Prevention

### Best Practices
1. **Testing**: Test all changes in development environment
2. **Monitoring**: Set up comprehensive monitoring and alerts
3. **Backups**: Maintain regular automated backups
4. **Documentation**: Keep deployment documentation updated
5. **Versioning**: Use proper version control for all changes

### Regular Maintenance
1. **Updates**: Keep dependencies updated
2. **Monitoring**: Review logs regularly
3. **Performance**: Monitor and optimize resource usage
4. **Security**: Rotate API keys and passwords regularly

---

For issues not covered in this guide, please check the [main deployment guide](DEPLOY_TO_RENDER.md) or open an issue in the GitHub repository.