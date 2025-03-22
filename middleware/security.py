from functools import wraps
from flask import request, redirect, url_for, current_app, Response, g
from urllib.parse import urlparse
import jwt
from datetime import datetime, timezone

def configure_security_headers():
    """Configure security headers for the application."""
    def security_headers(response: Response) -> Response:
        headers = current_app.config['SECURITY_HEADERS']
        for header, value in headers.items():
            if header not in response.headers:
                response.headers[header] = value
        return response
    return security_headers

def validate_jwt_token():
    """Middleware to validate JWT tokens."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return {'error': 'Missing or invalid authorization header'}, 401

            token = auth_header.split(' ')[1]
            try:
                # Decode token without verification first to check expiration
                unverified_payload = jwt.decode(
                    token, 
                    options={'verify_signature': False}
                )
                
                # Check token expiration
                exp = datetime.fromtimestamp(unverified_payload['exp'], timezone.utc)
                if datetime.now(timezone.utc) >= exp:
                    return {'error': 'Token has expired'}, 401

                # Verify token signature
                payload = jwt.decode(
                    token,
                    current_app.config['SUPABASE_JWT_SECRET'],
                    algorithms=['HS256']
                )
                
                # Store user info in Flask g object
                g.user_id = payload['sub']
                g.user_role = payload.get('role', 'free')
                g.user_email = payload.get('email')
                
                return f(*args, **kwargs)
            except jwt.ExpiredSignatureError:
                return {'error': 'Token has expired'}, 401
            except jwt.InvalidTokenError:
                return {'error': 'Invalid token'}, 401
        return decorated_function
    return decorator

def role_required(required_role):
    """Decorator to check if user has required role."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_role = getattr(g, 'user_role', None)
            if not user_role:
                return redirect(url_for('login'))
            
            roles = {
                'free': 0,
                'premium': 1,
                'admin': 2
            }
            
            if roles.get(user_role, -1) < roles.get(required_role, 999):
                return {'error': 'Insufficient permissions'}, 403
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_csrf_token():
    """Middleware to validate CSRF tokens."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
                csrf_token = request.headers.get('X-CSRF-Token')
                if not csrf_token:
                    return {'error': 'CSRF token missing'}, 400
                
                if not current_app.csrf.validate_csrf(csrf_token):
                    return {'error': 'Invalid CSRF token'}, 400
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_cors():
    """Middleware to validate CORS requests."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            origin = request.headers.get('Origin')
            if origin:
                parsed_origin = urlparse(origin)
                allowed_origins = current_app.config.get('CORS_ORIGINS', [])
                
                if parsed_origin.netloc not in allowed_origins:
                    return {'error': 'Origin not allowed'}, 403
                    
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def rate_limit_by_ip(limit_string):
    """Rate limiting decorator that uses client IP."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_app.config.get('RATELIMIT_ENABLED', True):
                return f(*args, **kwargs)
                
            key = f"{request.remote_addr}:{request.endpoint}"
            limiter = current_app.extensions['limiter']
            
            if not limiter.test(key, limit_string):
                return {'error': 'Rate limit exceeded. Try again later.'}, 429
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator