-- Enable necessary extensions
create extension if not exists "uuid-ossp";
create extension if not exists "pgcrypto";

-- Create custom types
create type user_role as enum ('free', 'premium', 'admin');
create type subscription_status as enum ('active', 'canceled', 'expired');

-- Create users table (extends auth.users)
create table public.user_profiles (
    id uuid references auth.users primary key,
    username varchar(64) unique not null check (char_length(username) >= 3),
    email varchar(255) not null check (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    role user_role not null default 'free',
    totp_secret varchar(32),
    failed_login_attempts integer default 0,
    locked_until timestamp with time zone,
    is_active boolean default true,
    remember_token varchar(255),
    created_at timestamp with time zone default now(),
    updated_at timestamp with time zone default now()
);

-- Create session tracking table
create table public.user_sessions (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references public.user_profiles not null,
    device_id varchar(255),
    ip_address inet,
    user_agent text,
    last_active timestamp with time zone default now(),
    expires_at timestamp with time zone not null,
    is_2fa_verified boolean default false,
    remember_me boolean default false,
    created_at timestamp with time zone default now()
);

-- Create password reset tokens table
create table public.password_reset_tokens (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references public.user_profiles not null,
    token varchar(255) not null,
    created_at timestamp with time zone default now(),
    expires_at timestamp with time zone not null,
    used_at timestamp with time zone
);

-- Create user device table for 2FA remember-device
create table public.user_devices (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references public.user_profiles not null,
    device_id varchar(255) not null,
    device_name text,
    is_trusted boolean default false,
    last_used_at timestamp with time zone default now(),
    expires_at timestamp with time zone not null,
    created_at timestamp with time zone default now()
);

-- Create audit log table
create table public.audit_logs (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references public.user_profiles,
    action varchar(50) not null,
    details jsonb,
    ip_address inet,
    user_agent text,
    created_at timestamp with time zone default now()
);

-- Enable Row Level Security
alter table public.user_profiles enable row level security;
alter table public.user_sessions enable row level security;
alter table public.password_reset_tokens enable row level security;
alter table public.user_devices enable row level security;
alter table public.audit_logs enable row level security;

-- Create RLS Policies

-- User Profiles policies
create policy "Users can view their own profile"
    on public.user_profiles for select
    using (auth.uid() = id);

create policy "Users can update their own profile"
    on public.user_profiles for update
    using (auth.uid() = id)
    with check (
        case
            when auth.jwt()->>'role' = 'admin' then true
            else (role = 'free' or role = 'premium') -- Non-admins can't promote themselves
        end
    );

create policy "Admins can view all profiles"
    on public.user_profiles for select
    using (auth.jwt()->>'role' = 'admin');

create policy "Admins can update all profiles"
    on public.user_profiles for update
    using (auth.jwt()->>'role' = 'admin');

-- User Sessions policies
create policy "Users can view their own sessions"
    on public.user_sessions for select
    using (auth.uid() = user_id);

create policy "Users can delete their own sessions"
    on public.user_sessions for delete
    using (auth.uid() = user_id);

create policy "Admins can view all sessions"
    on public.user_sessions for select
    using (auth.jwt()->>'role' = 'admin');

-- Password Reset Tokens policies
create policy "Users can view their own reset tokens"
    on public.password_reset_tokens for select
    using (auth.uid() = user_id);

create policy "Admins can view all reset tokens"
    on public.password_reset_tokens for select
    using (auth.jwt()->>'role' = 'admin');

-- User Devices policies
create policy "Users can view their own devices"
    on public.user_devices for select
    using (auth.uid() = user_id);

create policy "Users can delete their own devices"
    on public.user_devices for delete
    using (auth.uid() = user_id);

create policy "Admins can view all devices"
    on public.user_devices for select
    using (auth.jwt()->>'role' = 'admin');

-- Audit Logs policies
create policy "Users can view their own audit logs"
    on public.audit_logs for select
    using (auth.uid() = user_id);

create policy "Admins can view all audit logs"
    on public.audit_logs for select
    using (auth.jwt()->>'role' = 'admin');

-- Create function to automatically update updated_at
create or replace function update_updated_at_column()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

-- Create triggers for updated_at
create trigger set_timestamp
    before update on public.user_profiles
    for each row
    execute procedure update_updated_at_column();

-- Create indexes for performance
create index idx_user_profiles_username on public.user_profiles(username);
create index idx_user_profiles_email on public.user_profiles(email);
create index idx_user_sessions_user_id on public.user_sessions(user_id);
create index idx_user_sessions_device_id on public.user_sessions(device_id);
create index idx_password_reset_tokens_user_id on public.password_reset_tokens(user_id);
create index idx_password_reset_tokens_token on public.password_reset_tokens(token);
create index idx_user_devices_user_id on public.user_devices(user_id);
create index idx_user_devices_device_id on public.user_devices(device_id);
create index idx_audit_logs_user_id on public.audit_logs(user_id);
create index idx_audit_logs_action on public.audit_logs(action);

-- Set up notifications for security events
create or replace function notify_failed_login()
returns trigger as $$
begin
    if new.failed_login_attempts >= 5 then
        perform pg_notify(
            'security_events',
            json_build_object(
                'event', 'account_locked',
                'user_id', new.id,
                'username', new.username
            )::text
        );
    end if;
    return new;
end;
$$ language plpgsql;

create trigger notify_failed_login_trigger
    after update of failed_login_attempts on public.user_profiles
    for each row
    execute procedure notify_failed_login();