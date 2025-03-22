# Security Implementation Guide

This document outlines the security measures, configuration requirements, and testing procedures for the BallBetz authentication system.

## Architecture Overview

The authentication system uses Supabase for secure user management and PostgreSQL Row Level Security (RLS) policies. It implements:

- JWT-based authentication
- Role-based access control (RBAC)
- Two-factor authentication (2FA)
- Rate limiting
- CSRF protection
- Secure session management
- Password reset functionality
- Audit logging
- Row Level Security (RLS)

## Configuration Requirements

### Environment Variables

Required environment variables (see `.env.example`):

```bash
# Flask Settings
FLASK_APP=app.py
FLASK_ENV=development  # Change to 'production' in production
FLASK_SECRET_KEY=your-secret-key-here

# Supabase Settings
SUPABASE_URL=your-supabase-project-url
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_SERVICE_KEY=your-supabase-service-key
SUPABASE_JWT_SECRET=your-supabase-jwt-secret
```

### Security Settings

Default security settings (configurable in `config.py`):

```python
PASSWORD_RESET_TIMEOUT = 3600  # 1 hour
FAILED_LOGIN_TIMEOUT = 900     # 15 minutes
MAX_LOGIN_ATTEMPTS = 5
BCRYPT_LOG_ROUNDS = 12
MIN_PASSWORD_LENGTH = 12
```

## Password Requirements

Passwords must meet the following requirements:

- Minimum 12 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character
- Not contain common words or patterns

## Role-Based Access Control

Three user roles are supported:

1. **Free**: Basic access
   - Can view own profile
   - Can access basic features
   - Cannot access premium features

2. **Premium**: Enhanced access
   - All Free user permissions
   - Access to premium features
   - Enhanced rate limits

3. **Admin**: Full access
   - All Premium user permissions
   - Access to admin dashboard
   - Can manage user accounts
   - Can view audit logs

## Two-Factor Authentication (2FA)

2FA implementation uses TOTP (Time-based One-Time Password):

- Optional for Free/Premium users
- Required for Admin users
- 30-day remember device option
- Backup codes provided
- Rate limited verification attempts

## Rate Limiting

Default rate limits:

```python
# Login attempts
'5 per minute'     # Per IP address
'100 per day'      # Per IP address

# Password reset requests
'3 per hour'       # Per IP address
'5 per day'        # Per email address

# API endpoints
'200 per day'      # Standard limit
'1000 per day'     # Premium users
```

## Row Level Security (RLS)

RLS policies are defined in `schema.sql` and enforce:

- Users can only access their own data
- Admins can access all data
- Automatic filtering of queries
- Prevention of unauthorized modifications

## Session Management

Session security features:

- JWT-based authentication
- Secure cookie settings
- Session timeout
- Device tracking
- Concurrent session limits
- Remember-me functionality

## CSRF Protection

CSRF protection includes:

- CSRF tokens required for state-changing operations
- Token rotation
- Secure token validation
- SameSite cookie settings

## Security Headers

Default security headers:

```python
SECURITY_HEADERS = {
    'X-Frame-Options': 'DENY',
    'X-Content-Type-Options': 'nosniff',
    'X-XSS-Protection': '1; mode=block',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Content-Security-Policy': "default-src 'self'..."
}
```

## Audit Logging

Events logged include:

- Login attempts (successful and failed)
- Password changes
- Role changes
- 2FA enablement/disablement
- Password reset requests
- Admin actions

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run security-specific tests
pytest tests/unit/middleware/test_security.py
pytest tests/unit/database/test_init.py
pytest tests/unit/database/test_supabase_db.py
pytest tests/integration/test_auth.py
```

### Test Coverage

Test coverage should include:

- Password validation
- Role-based access control
- Rate limiting
- Session management
- CSRF protection
- 2FA functionality
- RLS policies
- Audit logging

## Security Best Practices

1. **Environment Variables**
   - Never commit sensitive values
   - Use strong, unique values in production
   - Rotate secrets regularly

2. **Database Security**
   - Enable RLS on all tables
   - Use prepared statements
   - Regular security audits
   - Backup procedures

3. **Authentication**
   - Enforce password complexity
   - Implement account lockouts
   - Rate limit authentication attempts
   - Secure session management

4. **Authorization**
   - Principle of least privilege
   - Role-based access control
   - Row Level Security
   - Regular permission audits

5. **Monitoring**
   - Comprehensive audit logging
   - Alert on suspicious activity
   - Regular security reviews
   - Performance monitoring

## Deployment Checklist

- [ ] Set secure environment variables
- [ ] Enable production configurations
- [ ] Verify RLS policies
- [ ] Configure rate limiting
- [ ] Set up SSL/TLS
- [ ] Configure security headers
- [ ] Enable audit logging
- [ ] Test all security features
- [ ] Review access controls
- [ ] Set up monitoring

## Emergency Response

1. **Security Breach**
   - Revoke compromised tokens
   - Reset affected passwords
   - Review audit logs
   - Notify affected users
   - Document incident

2. **System Issues**
   - Monitor error rates
   - Alert on anomalies
   - Automatic rollbacks
   - Incident reporting

## Maintenance

Regular security maintenance includes:

- Dependency updates
- Security patches
- Configuration reviews
- Access control audits
- Performance monitoring
- Security testing
- Documentation updates