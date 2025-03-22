-- Enable necessary extensions
create extension if not exists "uuid-ossp";
create extension if not exists "pgcrypto";

-- Create custom types
create type user_role as enum ('free', 'premium', 'admin', 'enterprise');
create type subscription_status as enum ('active', 'canceled', 'expired');
create type rewards_tier as enum ('rookie', 'all_star', 'mvp');

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
    updated_at timestamp with time zone default now(),
    rewards_tier rewards_tier default 'rookie',
    rewards_points integer default 0,
    data_sharing_level integer default 0
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

-- Create rewards tables
create table public.rewards_points_history (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references public.user_profiles not null,
    points integer not null,
    action varchar(50) not null,
    description text,
    created_at timestamp with time zone default now()
);

create table public.rewards_redemption_history (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references public.user_profiles not null,
    points_spent integer not null,
    reward_type varchar(50) not null,
    reward_description text,
    created_at timestamp with time zone default now()
);

create table public.rewards_tier_benefits (
    id uuid default uuid_generate_v4() primary key,
    tier rewards_tier not null,
    benefit_name varchar(100) not null,
    benefit_description text,
    created_at timestamp with time zone default now(),
    updated_at timestamp with time zone default now()
);

create table public.rewards_tier_requirements (
    id uuid default uuid_generate_v4() primary key,
    tier rewards_tier not null,
    min_points integer not null,
    min_data_sharing_level integer not null,
    subscription_discount decimal(5,2) not null,
    created_at timestamp with time zone default now(),
    updated_at timestamp with time zone default now()
);

create table public.user_data_sharing_settings (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references public.user_profiles not null,
    setting_name varchar(100) not null,
    is_enabled boolean default false,
    created_at timestamp with time zone default now(),
    updated_at timestamp with time zone default now(),
    unique(user_id, setting_name)
);

-- Enable Row Level Security
alter table public.user_profiles enable row level security;
alter table public.user_sessions enable row level security;
alter table public.password_reset_tokens enable row level security;
alter table public.user_devices enable row level security;
alter table public.audit_logs enable row level security;

-- Enable RLS for rewards tables
alter table public.rewards_points_history enable row level security;
alter table public.rewards_redemption_history enable row level security;
alter table public.rewards_tier_benefits enable row level security;
alter table public.user_data_sharing_settings enable row level security;

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

-- Rewards Points History policies
create policy "Users can view their own rewards points history"
    on public.rewards_points_history for select
    using (auth.uid() = user_id);

create policy "Admins can view all rewards points history"
    on public.rewards_points_history for select
    using (auth.jwt()->>'role' = 'admin');

-- Rewards Redemption History policies
create policy "Users can view their own redemption history"
    on public.rewards_redemption_history for select
    using (auth.uid() = user_id);

create policy "Admins can view all redemption history"
    on public.rewards_redemption_history for select
    using (auth.jwt()->>'role' = 'admin');

-- Rewards Tier Benefits policies
create policy "Anyone can view tier benefits"
    on public.rewards_tier_benefits for select
    using (true);

create policy "Admins can manage tier benefits"
    on public.rewards_tier_benefits for all
    using (auth.jwt()->>'role' = 'admin');

-- User Data Sharing Settings policies
create policy "Users can view their own data sharing settings"
    on public.user_data_sharing_settings for select
    using (auth.uid() = user_id);

create policy "Users can update their own data sharing settings"
    on public.user_data_sharing_settings for update
    using (auth.uid() = user_id);

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

create trigger set_rewards_tier_benefits_timestamp
    before update on public.rewards_tier_benefits
    for each row
    execute procedure update_updated_at_column();

create trigger set_rewards_tier_requirements_timestamp
    before update on public.rewards_tier_requirements
    for each row
    execute procedure update_updated_at_column();

create trigger set_user_data_sharing_settings_timestamp
    before update on public.user_data_sharing_settings
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
create index idx_rewards_points_history_user_id on public.rewards_points_history(user_id);
create index idx_rewards_points_history_action on public.rewards_points_history(action);
create index idx_rewards_redemption_history_user_id on public.rewards_redemption_history(user_id);
create index idx_rewards_redemption_history_reward_type on public.rewards_redemption_history(reward_type);
create index idx_rewards_tier_benefits_tier on public.rewards_tier_benefits(tier);
create index idx_rewards_tier_requirements_tier on public.rewards_tier_requirements(tier);
create index idx_user_data_sharing_settings_user_id on public.user_data_sharing_settings(user_id);
create index idx_user_data_sharing_settings_setting_name on public.user_data_sharing_settings(setting_name);

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

-- Insert default rewards tier benefits
insert into public.rewards_tier_benefits (tier, benefit_name, benefit_description)
values
    ('rookie', 'Custom Avatar Badges', 'Unique avatar badges to show your Rookie status'),
    ('rookie', 'Basic Performance Insights', 'Access to basic performance metrics and insights'),
    ('all_star', 'Priority Lineup Processing', 'Your lineup optimization requests are processed with higher priority'),
    ('all_star', 'Exclusive Betting Guides', 'Access to exclusive betting strategy guides'),
    ('all_star', 'Community Leaderboard Access', 'See how you rank against other BallBetz users'),
    ('mvp', 'Beta Feature Access', 'Early access to new features before they are released to everyone'),
    ('mvp', 'Monthly Strategy Sessions', 'Monthly virtual strategy sessions with betting experts'),
    ('mvp', 'Personalized Betting Insights', 'AI-powered personalized betting recommendations'),
    ('mvp', 'VIP Discord Access', 'Access to the exclusive VIP Discord community');

-- Insert default rewards tier requirements
insert into public.rewards_tier_requirements (tier, min_points, min_data_sharing_level, subscription_discount)
values
    ('rookie', 0, 1, 10.00),
    ('all_star', 1000, 2, 25.00),
    ('mvp', 5000, 3, 40.00);

-- Create default data sharing settings
create or replace function create_default_data_sharing_settings()
returns trigger as $$
begin
    insert into public.user_data_sharing_settings (user_id, setting_name, is_enabled)
    values
        (new.id, 'profile_visibility', false),
        (new.id, 'betting_history', false),
        (new.id, 'favorite_teams', false),
        (new.id, 'betting_patterns', false),
        (new.id, 'platform_usage', false),
        (new.id, 'performance_stats', false);
    return new;
end;
$$ language plpgsql;

-- Create trigger to create default data sharing settings for new users
create trigger create_default_data_sharing_settings_trigger
    after insert on public.user_profiles
    for each row
    execute procedure create_default_data_sharing_settings();