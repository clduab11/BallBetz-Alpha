# StatShare Rewards™ System Documentation

## Overview

The StatShare Rewards™ program is a loyalty system that incentivizes users to share their data by providing tangible benefits. Users earn points through platform engagement and can redeem these points for various rewards. The system includes tiered membership levels with escalating benefits based on user participation and data sharing.

## Core Components

### 1. Rewards Tiers

The system has three membership tiers with increasing benefits:

| Tier | Requirements | Discount | Benefits |
|------|--------------|----------|----------|
| **Rookie** | 0+ points, Level 1+ data sharing | 10% off subscription | • Custom avatar badges<br>• Basic performance insights |
| **All-Star** | 1,000+ points, Level 2+ data sharing | 25% off subscription | • Priority lineup processing<br>• Exclusive betting guides<br>• Community leaderboard access |
| **MVP** | 5,000+ points, Level 3 data sharing | 40% off subscription | • Beta feature access<br>• Monthly strategy sessions<br>• Personalized betting insights<br>• VIP Discord access |

### 2. Points System

Users earn points through various platform activities:

| Action | Points Awarded |
|--------|----------------|
| Daily Login | 10 points |
| Profile Completion | 50 points |
| First Bet | 100 points |
| Winning Bet | 25 points |
| Refer a Friend | 200 points |
| Feedback Submission | 30 points |
| Contest Participation | 75 points |
| Social Share | 15 points |

### 3. Data Sharing Levels

The system has three data sharing levels that determine tier eligibility:

| Level | Description | Requirements |
|-------|-------------|--------------|
| **Level 1: Basic** | Minimal data sharing | 1-2 settings enabled |
| **Level 2: Moderate** | Moderate data sharing | 3-4 settings enabled |
| **Level 3: Comprehensive** | Comprehensive data sharing | 5-6 settings enabled |

### 4. Redemption Options

Users can redeem points for various rewards:

| Reward | Points Required | Description |
|--------|----------------|-------------|
| Subscription Discount | 1,000 points | 10% off next month's subscription |
| Premium Feature Access | 500 points | 1 week of premium features |
| Exclusive Content | 300 points | Access to expert strategy guides |
| Custom Analysis Report | 2,000 points | Personalized betting analysis |

## Database Schema

The rewards system uses the following database tables:

### User Profile Extensions

The `user_profiles` table is extended with the following fields:

```sql
rewards_tier rewards_tier default 'rookie',
rewards_points integer default 0,
data_sharing_level integer default 0
```

### Rewards-Specific Tables

```sql
-- Points history
create table public.rewards_points_history (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references public.user_profiles not null,
    points integer not null,
    action varchar(50) not null,
    description text,
    created_at timestamp with time zone default now()
);

-- Redemption history
create table public.rewards_redemption_history (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references public.user_profiles not null,
    points_spent integer not null,
    reward_type varchar(50) not null,
    reward_description text,
    created_at timestamp with time zone default now()
);

-- Tier benefits
create table public.rewards_tier_benefits (
    id uuid default uuid_generate_v4() primary key,
    tier rewards_tier not null,
    benefit_name varchar(100) not null,
    benefit_description text,
    created_at timestamp with time zone default now(),
    updated_at timestamp with time zone default now()
);

-- Tier requirements
create table public.rewards_tier_requirements (
    id uuid default uuid_generate_v4() primary key,
    tier rewards_tier not null,
    min_points integer not null,
    min_data_sharing_level integer not null,
    subscription_discount decimal(5,2) not null,
    created_at timestamp with time zone default now(),
    updated_at timestamp with time zone default now()
);

-- Data sharing settings
create table public.user_data_sharing_settings (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references public.user_profiles not null,
    setting_name varchar(100) not null,
    is_enabled boolean default false,
    created_at timestamp with time zone default now(),
    updated_at timestamp with time zone default now(),
    unique(user_id, setting_name)
);
```

## API Endpoints

### User-Facing Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rewards/dashboard` | GET | View rewards dashboard with tier status, points, and benefits |
| `/rewards/settings` | GET/POST | View and update data sharing settings |
| `/rewards/redeem` | GET/POST | View available rewards and redeem points |
| `/rewards/history` | GET | View points earning and redemption history |

### Internal API Methods

#### User Model Methods

| Method | Description |
|--------|-------------|
| `get_rewards_tier()` | Get the user's current rewards tier |
| `get_rewards_points()` | Get the user's current rewards points |
| `get_data_sharing_level()` | Get the user's current data sharing level |
| `get_subscription_discount()` | Get the user's subscription discount based on rewards tier |
| `add_rewards_points(points, action, description)` | Add rewards points to the user's account |
| `redeem_rewards_points(points, reward_type, reward_description)` | Redeem rewards points for a reward |
| `check_tier_upgrade()` | Check if user qualifies for a tier upgrade |
| `check_tier_downgrade()` | Check if user should be downgraded to a lower tier |
| `update_data_sharing_settings(settings)` | Update user's data sharing settings |
| `get_data_sharing_settings()` | Get user's data sharing settings |

#### Rewards Class Methods

| Method | Description |
|--------|-------------|
| `get_tier_benefits(tier)` | Get benefits for a specific tier or all tiers |
| `get_tier_requirements(tier)` | Get requirements for a specific tier or all tiers |
| `get_points_history(user_id)` | Get points history for a user |
| `get_redemption_history(user_id)` | Get redemption history for a user |
| `award_points_for_action(user_id, action)` | Award points to a user based on a predefined action |

## Implementation Details

### Data Sharing Settings

The system tracks the following data sharing settings:

1. **Profile Visibility**: Allow other users to see your profile information
2. **Betting History**: Share your anonymized betting history to improve platform predictions
3. **Favorite Teams**: Share your favorite teams to receive personalized recommendations
4. **Betting Patterns**: Share your betting patterns to help improve the platform algorithms
5. **Platform Usage**: Share your platform usage data to help us improve the user experience
6. **Performance Stats**: Share your performance statistics to contribute to community insights

### Tier Calculation Logic

The system automatically calculates the appropriate tier for each user based on two factors:

1. **Points Threshold**: User must have accumulated the minimum number of points required for the tier
2. **Data Sharing Level**: User must have enabled enough data sharing settings to reach the required level

Tier upgrades happen immediately when both conditions are met. Tier downgrades happen when a user falls below either threshold (e.g., by redeeming points or disabling data sharing settings).

### Points Awarding System

Points are awarded automatically for various user actions:

- **Recurring Actions**: Actions like daily login can be performed repeatedly
- **One-Time Actions**: Actions like profile completion can only be performed once
- **Achievement-Based**: Actions like winning a bet are triggered by specific achievements

### Security Considerations

The rewards system implements the following security measures:

- **Row-Level Security**: All rewards tables use Postgres RLS to ensure users can only access their own data
- **Audit Logging**: All significant actions (tier changes, point redemptions) are logged in the audit system
- **Data Anonymization**: Shared data is anonymized before being used for platform improvements
- **Granular Permissions**: Users have complete control over what data is shared

## User Interface

The rewards system includes the following UI components:

1. **Rewards Dashboard**: Main hub showing tier status, points, and available benefits
2. **Data Sharing Settings**: Interface for managing what data is shared
3. **Redemption Center**: Interface for redeeming points for rewards
4. **History View**: Interface for viewing points earning and redemption history

## Testing

The rewards system includes comprehensive testing:

- **Unit Tests**: Test individual methods in the User and Rewards classes
- **Integration Tests**: Test the interaction between components and database operations
- **Route Tests**: Test the HTTP endpoints and form submissions

## Future Enhancements

Planned enhancements for the rewards system include:

1. **Rewards Marketplace**: Allow users to purchase and trade rewards
2. **Seasonal Promotions**: Special limited-time rewards and point multipliers
3. **Team Rewards**: Group rewards for users who form betting teams
4. **Achievement System**: Badges and special rewards for achieving specific milestones
5. **Referral Tiers**: Enhanced rewards for users who refer multiple friends

## Conclusion

The StatShare Rewards™ system provides a comprehensive loyalty program that incentivizes data sharing while maintaining user privacy and control. By offering tangible benefits for participation, the system creates a win-win scenario where users receive discounts and exclusive features while the platform gains valuable data to improve its predictions and services.