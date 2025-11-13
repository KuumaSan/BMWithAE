# Production Deployment Guide

This guide explains how to deploy BMWithAE to a production server for long-term operation.

## 🚀 Quick Start

### Option 1: Using the Provided Scripts (Recommended)

**Linux/Mac:**
```bash
cd BMWithAE/backend
chmod +x start_production.sh
./start_production.sh
```

**Windows:**
```cmd
cd BMWithAE\backend
start_production.bat
```

---

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), Windows Server, or macOS
- **Python**: 3.8 or higher
- **Memory**: Minimum 2GB RAM (4GB+ recommended)
- **Disk**: 10GB+ free space
- **Network**: Open port 5001 (or your configured port)

### Software Requirements
- Python 3.8+
- pip
- Virtual environment (recommended)
- (Optional) Nginx for reverse proxy
- (Optional) Supervisor or systemd for process management

---

## 🔧 Production Setup

### Step 1: Clone and Install

```bash
# Clone repository
git clone https://github.com/yourusername/BMWithAE.git
cd BMWithAE

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

Create a `.env` file in the `backend/` directory:

```bash
# Environment
FLASK_ENV=production

# Server
FLASK_HOST=0.0.0.0
FLASK_PORT=5001

# Optional: Database, logging, etc.
```

### Step 3: Test Production Server

```bash
cd backend

# Test with Gunicorn (Linux/Mac)
gunicorn --bind 0.0.0.0:5001 --workers 4 wsgi:app

# Test with Waitress (Cross-platform)
waitress-serve --host=0.0.0.0 --port=5001 wsgi:app
```

Visit `http://your-server-ip:5001/api/config` to verify the backend is running.

---

## 🔄 Process Management

### Option A: Using Systemd (Linux - Recommended)

1. **Copy and edit the service file:**
```bash
sudo cp deployment/bmwithae.service /etc/systemd/system/
sudo nano /etc/systemd/system/bmwithae.service
```

2. **Update paths in the service file:**
   - `WorkingDirectory`: `/path/to/BMWithAE/backend`
   - `ExecStart`: `/path/to/venv/bin/gunicorn ...`
   - `User` and `Group`: Your application user

3. **Enable and start the service:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable bmwithae
sudo systemctl start bmwithae
sudo systemctl status bmwithae
```

4. **Useful commands:**
```bash
sudo systemctl stop bmwithae      # Stop service
sudo systemctl restart bmwithae   # Restart service
sudo systemctl status bmwithae    # Check status
sudo journalctl -u bmwithae -f    # View logs
```

### Option B: Using Supervisor

1. **Install Supervisor:**
```bash
sudo apt-get install supervisor  # Ubuntu/Debian
sudo yum install supervisor      # CentOS/RHEL
```

2. **Copy and edit configuration:**
```bash
sudo cp deployment/supervisor.conf /etc/supervisor/conf.d/bmwithae.conf
sudo nano /etc/supervisor/conf.d/bmwithae.conf
```

3. **Update paths in the configuration file**

4. **Start the application:**
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start bmwithae
sudo supervisorctl status bmwithae
```

### Option C: Using PM2 (Alternative)

```bash
# Install PM2
npm install -g pm2

# Start application
cd backend
pm2 start "gunicorn --bind 0.0.0.0:5001 --workers 4 wsgi:app" --name bmwithae

# Useful commands
pm2 status                # Check status
pm2 logs bmwithae        # View logs
pm2 restart bmwithae     # Restart
pm2 stop bmwithae        # Stop
pm2 startup              # Auto-start on boot
```

---

## 🌐 Nginx Reverse Proxy Setup

### Benefits
- SSL/TLS termination
- Load balancing
- Static file serving
- Better security
- Domain name support

### Installation

1. **Install Nginx:**
```bash
sudo apt-get install nginx  # Ubuntu/Debian
```

2. **Copy and edit configuration:**
```bash
sudo cp deployment/nginx.conf /etc/nginx/sites-available/bmwithae
sudo nano /etc/nginx/sites-available/bmwithae
```

3. **Update configuration:**
   - Replace `your-domain.com` with your domain or server IP
   - Update `/path/to/BMWithAE/frontend` with actual path

4. **Enable site:**
```bash
sudo ln -s /etc/nginx/sites-available/bmwithae /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl restart nginx
```

5. **Access your application:**
   - Frontend: `http://your-domain.com`
   - Backend API: `http://your-domain.com/api/`

---

## 🔒 Security Considerations

### 1. Firewall Configuration

```bash
# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# If not using Nginx, allow backend port
sudo ufw allow 5001/tcp

# Enable firewall
sudo ufw enable
```

### 2. SSL/TLS Certificate (HTTPS)

Using Let's Encrypt (free):

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is configured automatically
sudo certbot renew --dry-run
```

### 3. Environment Variables

Never commit sensitive data to git. Use `.env` files:

```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo "backend/.env" >> .gitignore
```

### 4. Disable Debug Mode

Ensure `DEBUG = False` in production. The configuration automatically sets this when `FLASK_ENV=production`.

---

## 📊 Monitoring and Logging

### Application Logs

Logs are stored in `backend/logs/`:
- `access.log`: HTTP access logs
- `error.log`: Application errors
- `supervisor.log`: Supervisor logs (if using Supervisor)

### View Logs

```bash
# Gunicorn logs
tail -f backend/logs/access.log
tail -f backend/logs/error.log

# Systemd logs
sudo journalctl -u bmwithae -f

# Nginx logs
sudo tail -f /var/log/nginx/bmwithae_access.log
sudo tail -f /var/log/nginx/bmwithae_error.log
```

### Log Rotation

Configure logrotate for automatic log management:

```bash
sudo nano /etc/logrotate.d/bmwithae
```

Add:
```
/path/to/BMWithAE/backend/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
}
```

---

## 🔄 Updates and Maintenance

### Deploying Updates

```bash
# 1. Pull latest code
cd BMWithAE
git pull origin main

# 2. Activate virtual environment
source venv/bin/activate

# 3. Update dependencies
pip install -r requirements.txt --upgrade

# 4. Restart application
# Using systemd:
sudo systemctl restart bmwithae

# Using supervisor:
sudo supervisorctl restart bmwithae

# Using PM2:
pm2 restart bmwithae
```

### Database Backups (if applicable)

```bash
# Backup uploads and results
tar -czf backup-$(date +%Y%m%d).tar.gz backend/uploads backend/results backend/logs
```

---

## 🐛 Troubleshooting

### Backend Not Starting

1. **Check logs:**
```bash
sudo journalctl -u bmwithae -n 50
# or
cat backend/logs/error.log
```

2. **Verify Python dependencies:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

3. **Test manually:**
```bash
cd backend
python app.py
```

### Port Already in Use

```bash
# Find process using port 5001
sudo lsof -i :5001
# or
sudo netstat -tulpn | grep 5001

# Kill process
sudo kill -9 <PID>
```

### Permission Issues

```bash
# Fix file permissions
sudo chown -R www-data:www-data /path/to/BMWithAE
sudo chmod -R 755 /path/to/BMWithAE

# Ensure log directories are writable
sudo chmod -R 755 backend/logs backend/uploads backend/results
```

### Memory Issues

If running out of memory, reduce worker count:

```bash
# Edit service or script
# Change --workers 4 to --workers 2
```

---

## 📈 Performance Optimization

### 1. Worker Configuration

Rule of thumb: `workers = 2-4 × CPU_CORES`

```bash
# Check CPU cores
nproc

# Adjust workers in start_production.sh or service file
```

### 2. Caching

Consider adding Redis for session/cache management:

```bash
pip install redis flask-caching
```

### 3. Database Optimization

If you add a database later, consider:
- Connection pooling
- Query optimization
- Indexing

---

## 📞 Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/BMWithAE/issues
- Documentation: See README.md

---

## ✅ Deployment Checklist

Before going live:

- [ ] Python dependencies installed
- [ ] Environment variables configured
- [ ] DEBUG mode disabled
- [ ] Process manager configured (systemd/supervisor)
- [ ] Nginx reverse proxy set up (optional)
- [ ] Firewall rules configured
- [ ] SSL certificate installed (for HTTPS)
- [ ] Log rotation configured
- [ ] Backup strategy in place
- [ ] Monitoring set up
- [ ] Application tested thoroughly

---

**Congratulations! Your BMWithAE application is now running in production. 🎉**

