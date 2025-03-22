# BallBetz Security Implementation

## Overview

This repository contains a comprehensive security implementation for the BallBetz application, focusing on secure authentication, authorization, and data protection using Supabase and Flask.

### Key Features

- JWT-based authentication with Supabase
- Role-based access control (RBAC)
- Two-factor authentication (2FA)
- Row Level Security (RLS) with Supabase
- Rate limiting and brute force protection
- Secure session management
- CSRF protection
- Comprehensive security testing suite
- Audit logging

## Architecture

The security implementation consists of several key components:

```
├── app.py                 # Main application file
├── config.py             # Security configuration
├── models.py             # User model with security features
├── forms.py              # Secure form handling
├── schema.sql           # Database schema with RLS policies
├── database/            # Database interface
│   ├── __init__.py     # Supabase initialization
│   └── supabase_db.py  # Database operations
├── middleware/          # Security middleware
│   └── security.py     # Security handlers
├── tests/              # Test suite
│   ├── unit/          # Unit tests
│   └── integration/   # Integration tests
└── docs/              # Documentation
    ├── SECURITY.md    # Security implementation details
    └── TEST_SPEC.md   # Test specifications
```

## Setup

### Prerequisites

- Python 3.8+
- Supabase account and project
- PostgreSQL 12+
- Flask 3.0+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ballbetz-alpha.git
cd ballbetz-alpha
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create environment file:
```bash
cp .env.example .env
```

4. Configure environment variables:
```bash
# Required environment variables in .env
FLASK_APP=app.py
FLASK_ENV=development
FLASK_SECRET_KEY=your-secret-key
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key
SUPABASE_SERVICE_KEY=your-service-key
SUPABASE_JWT_SECRET=your-jwt-secret
```

5. Initialize database:
```bash
# Apply database schema with RLS policies
psql -h your-supabase-host -d postgres -U postgres -f schema.sql
```

### Security Configuration

1. Configure security settings in `config.py`:
```python
RATELIMIT_DEFAULT = "200 per day"
PASSWORD_RESET_TIMEOUT = 3600
MAX_LOGIN_ATTEMPTS = 5
```

2. Enable security features in Supabase:
- Enable Row Level Security
- Configure auth providers
- Set up email templates
- Configure security policies

## Testing

### Running Tests

1. Unit Tests:
```bash
pytest tests/unit/
```

2. Integration Tests:
```bash
pytest tests/integration/
```

3. Security Tests:
```bash
pytest tests/unit/middleware/test_security.py
pytest tests/unit/database/test_supabase_db.py
pytest tests/integration/test_auth.py
```

4. Coverage Report:
```bash
pytest --cov=app --cov-report=html tests/
```

### Test Documentation

- Detailed test specifications: [TEST_SPEC.md](docs/TEST_SPEC.md)
- Security implementation details: [SECURITY.md](docs/SECURITY.md)

## Security Features

### Authentication

- JWT-based authentication
- Two-factor authentication
- Password complexity requirements
- Account lockout protection
- Rate limiting
- Remember-me functionality

### Authorization

- Role-based access control
- Row Level Security policies
- Secure session management
- CSRF protection
- Security headers

### Audit Logging

- Authentication events
- Authorization events
- Security events
- User actions
- System events

## Deployment

### Production Configuration

1. Set secure environment variables
2. Enable production mode
3. Configure SSL/TLS
4. Set up monitoring
5. Enable audit logging

### Security Checklist

- [ ] Secure environment variables configured
- [ ] Production settings enabled
- [ ] SSL/TLS certificates installed
- [ ] Rate limiting configured
- [ ] RLS policies verified
- [ ] Audit logging enabled
- [ ] Monitoring set up
- [ ] Backups configured
- [ ] Security headers verified
- [ ] Tests passing

## Monitoring

### Security Monitoring

1. Monitor auth events:
```python
@app.after_request
def log_auth_events(response):
    if request.endpoint in ['login', 'register', 'reset_password']:
        log_security_event(request, response)
    return response
```

2. Check audit logs:
```sql
SELECT * FROM audit_logs 
WHERE action = 'failed_login' 
AND created_at > NOW() - INTERVAL '1 hour';
```

### Performance Monitoring

1. Monitor rate limits:
```python
@app.after_request
def monitor_rate_limits(response):
    if response.status_code == 429:
        log_rate_limit_exceeded(request)
    return response
```

2. Check performance metrics:
```sql
SELECT COUNT(*), action 
FROM audit_logs 
GROUP BY action 
ORDER BY count DESC;
```

## Contributing

### Security Guidelines

1. Follow security best practices
2. Write tests for new features
3. Update documentation
4. Maintain test coverage
5. Review security implications

### Pull Request Process

1. Update tests
2. Update documentation
3. Run security checks
4. Submit for review

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For security-related issues:
- Email: security@ballbetz.com
- Bug Bounty: https://ballbetz.com/security
- Security Policy: [SECURITY.md](docs/SECURITY.md)

For general support:
- Documentation: [docs/](docs/)
- Issues: GitHub Issues
- Wiki: GitHub Wiki