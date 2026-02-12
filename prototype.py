import pybaseball as pb
import pandas as pd
import numpy as np

# ========== BATTER CLASS ==========
class Batter:
    """Represents a batter with their metrics and weaknesses"""
    
    def __init__(self, batter_id, name, handedness):
        self.batter_id = batter_id
        self.name = name
        self.handedness = handedness  # 'L', 'R', or 'S' (switch)
        self.metrics = {}
        self.percentiles = {}
        self.weaknesses = []
        self.strengths = []
    
    def add_metrics(self, metrics_dict):
        """Add raw metric values"""
        self.metrics = metrics_dict
    
    def add_percentiles(self, percentile_dict):
        """Add percentile rankings"""
        self.percentiles = percentile_dict
    
    def identify_weaknesses(self, top_n=3):
        """Find this batter's worst percentiles"""
        weaknesses = []
        
        for metric_name, percentile in self.percentiles.items():
            if pd.isna(percentile):
                continue
            
            # Determine severity
            if percentile < 10:
                severity = 'CRITICAL'
            elif percentile < 20:
                severity = 'MAJOR'
            elif percentile < 30:
                severity = 'MODERATE'
            else:
                continue  # Not a weakness
            
            weaknesses.append({
                'metric': metric_name,
                'raw_value': round(self.metrics[metric_name], 3) if isinstance(self.metrics[metric_name], float) else self.metrics[metric_name],
                'percentile': round(percentile, 1),
                'severity': severity
            })
        
        # Sort by worst percentile (lowest first)
        weaknesses.sort(key=lambda x: x['percentile'])
        self.weaknesses = weaknesses[:top_n]
        return self.weaknesses
    
    def identify_strengths(self, top_n=3):
        """Find this batter's best percentiles"""
        strengths = []
        
        for metric_name, percentile in self.percentiles.items():
            if pd.isna(percentile):
                continue
            strengths.append({
                'metric': metric_name,
                'percentile': percentile
            })
        
        strengths.sort(key=lambda x: x['percentile'], reverse=True)
        self.strengths = strengths[:top_n]
        return self.strengths
    
    def generate_profile(self):
        """Human-readable scouting report"""
        profile = []
        profile.append("=" * 60)
        profile.append(f"BATTER: {self.name} ({self.handedness})")
        profile.append("=" * 60)
        profile.append("")
        profile.append("PRIMARY WEAKNESSES:")
        
        for i, w in enumerate(self.weaknesses, 1):
            profile.append(f"{i}. {w['severity']}: {w['metric']}")
            profile.append(f"   Value: {w['raw_value']} | Percentile: {w['percentile']}%")
            profile.append(f"   → {self._generate_strategy(w['metric'])}")
        
        profile.append("")
        profile.append("RELATIVE STRENGTHS:")
        
        for s in self.strengths:
            profile.append(f"   ✅ {s['metric']}: {s['percentile']:.0f}%")
        
        return "\n".join(profile)
    
    @staticmethod
    def _generate_strategy(metric_name):
        """Map metrics to pitching strategies"""
        strategies = {
            'xwOBACON': 'Weak contact even when squared—attack with velocity',
            'HardHit%': 'Can\'t generate power—challenge in zone',
            'Barrel%': 'Never squares it up—pitch to contact',
            'AvgEV': 'Overall weak contact—attack aggressively',
            'SweetSpot%': 'Can\'t lift the ball—elevate fastballs',
            'Chase%': 'Expands zone—bury breaking balls with 2 strikes',
            'Whiff%': 'Misses everything—throw chase pitches early',
            'FirstPitchSwing%': 'Predictably aggressive—first pitch breaking ball',
            'ZoneContact%': 'Misses hittable pitches—attack zone',
            'CalledStrike%': 'Takes meat pitches—throw strikes early',
            'xwOBA_HighFB': 'Can\'t handle elevation—live up',
            'xwOBA_LowAway': 'Can\'t extend—low/away sliders',
            'xwOBA_Inside': 'Jams easily—hard inside',
            'xwOBA_Breaking': 'Can\'t read spin—backfoot breaking balls',
            'xwOBA_Offspeed': 'Fooled by timing—changeups in fastball counts',
            'xwOBA_RISP': 'Presses with runners—expand zone early',
            'xwOBA_2Strikes': 'Defensive with two strikes—waste pitches',
            'xwOBA_SameSide': 'Platoon vulnerability—same-side reliever',
            'xwOBA_RepeatPitch': 'Can\'t adjust—repeat pitch sequences',
            'xwOBA_HittersCount': 'Even in advantage, can\'t punish—don\'t nibble'
        }
        return strategies.get(metric_name, 'Attack this weakness consistently')


# ========== DATA FETCHING ==========
batters = pb.batting_stats(2024, qual=200)
sample_batters = batters.sample(n=50, random_state=42)

batter_ids = sample_batters['IDfg'].tolist()
batter_names = sample_batters['Name'].tolist()

# Get player ID mapping
from pybaseball import playerid_reverse_lookup

# Convert FanGraphs IDs to MLBAM IDs
mlbam_ids = []
for fg_id in batter_ids:
    try:
        id_map = playerid_reverse_lookup([fg_id], key_type='fangraphs')
        if len(id_map) > 0:
            mlbam_ids.append(int(id_map['key_mlbam'].iloc[0]))
        else:
            mlbam_ids.append(None)
    except:
        mlbam_ids.append(None)

all_statcast = []

# Fetch Statcast data
for i, mlbam_id in enumerate(mlbam_ids):
    if mlbam_id is None:
        continue
    
    print(f"Fetching batter {i+1}/50: {batter_names[i]}")
    
    try:
        batter_data = pb.statcast_batter(
            start_dt='2024-03-28',
            end_dt='2024-10-01',
            player_id=mlbam_id
        )
        
        if len(batter_data) > 0:
            batter_data = batter_data.copy()
            batter_data['batter_name'] = batter_names[i]
            batter_data['fg_id'] = batter_ids[i]
            all_statcast.append(batter_data)
            
    except Exception as e:
        print(f"Error on {batter_names[i]}: {e}")
        continue

df = pd.concat(all_statcast, ignore_index=True)


# ========== METRICS CALCULATION ==========
class WeaknessMetrics:
    """Calculate all 20 metrics for any batter"""
    
    def __init__(self, df):
        self.df = df
        
    def calculate_all(self, batter_id, handedness=None):
        """Run all 20 metrics for one batter (optionally filtered by handedness)"""
        batter_df = self.df[self.df['batter'] == batter_id].copy()
        
        # Filter by handedness if specified (for switch hitters)
        if handedness:
            batter_df = batter_df[batter_df['stand'] == handedness]
        
        if len(batter_df) < 50:  # Minimum sample check
            return None
            
        metrics = {}
        
        # ========== CATEGORY A: CONTACT QUALITY ==========
        bb_events = batter_df[batter_df['estimated_woba_using_speedangle'].notna()]
        metrics['xwOBACON'] = bb_events['estimated_woba_using_speedangle'].mean()
        
        hard_hit = batter_df[batter_df['launch_speed'] >= 95]
        metrics['HardHit%'] = len(hard_hit) / len(bb_events) * 100 if len(bb_events) > 0 else 0
        
        metrics['Barrel%'] = batter_df['barrel'].mean() * 100 if 'barrel' in batter_df.columns else 0
        
        metrics['AvgEV'] = batter_df['launch_speed'].mean()
        
        sweet_spot = batter_df[(batter_df['launch_angle'] >= 8) & (batter_df['launch_angle'] <= 32)]
        metrics['SweetSpot%'] = len(sweet_spot) / len(bb_events) * 100 if len(bb_events) > 0 else 0
        
        # ========== CATEGORY B: PLATE DISCIPLINE ==========
        swing_descriptions = ['hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked', 'foul_tip', 'foul_bunt']
        
        outside_zone = batter_df[batter_df['zone'] > 9]
        swung_outside = outside_zone[outside_zone['description'].isin(swing_descriptions)]
        metrics['Chase%'] = len(swung_outside) / len(outside_zone) * 100 if len(outside_zone) > 0 else 0
        
        swings = batter_df[batter_df['description'].isin(swing_descriptions)]
        whiffs = batter_df[batter_df['description'].isin(['swinging_strike', 'swinging_strike_blocked'])]
        metrics['Whiff%'] = len(whiffs) / len(swings) * 100 if len(swings) > 0 else 0
        
        first_pitches = batter_df[batter_df['pitch_number'] == 1]
        first_swings = first_pitches[first_pitches['description'].isin(swing_descriptions)]
        metrics['FirstPitchSwing%'] = len(first_swings) / len(first_pitches) * 100 if len(first_pitches) > 0 else 0
        
        in_zone = batter_df[batter_df['zone'].between(1, 9)]
        in_zone_swings = in_zone[in_zone['description'].isin(swing_descriptions)]
        in_zone_contact = in_zone_swings[~in_zone_swings['description'].isin(['swinging_strike', 'swinging_strike_blocked'])]
        metrics['ZoneContact%'] = len(in_zone_contact) / len(in_zone_swings) * 100 if len(in_zone_swings) > 0 else 0
        
        non_swings = batter_df[~batter_df['description'].isin(swing_descriptions)]
        called_strikes = non_swings[non_swings['description'] == 'called_strike']
        metrics['CalledStrike%'] = len(called_strikes) / len(non_swings) * 100 if len(non_swings) > 0 else 0
        
        # ========== CATEGORY C: ZONE COVERAGE ==========
        high_fb = batter_df[
            (batter_df['zone'].isin([11, 12])) & 
            (batter_df['pitch_type'].isin(['FF', '4-Seam Fastball', 'FA']))
        ]
        metrics['xwOBA_HighFB'] = high_fb['estimated_woba_using_speedangle'].mean()
        
        low_away = batter_df[batter_df['zone'].isin([8, 9, 14])]
        metrics['xwOBA_LowAway'] = low_away['estimated_woba_using_speedangle'].mean()
        
        inside = batter_df[batter_df['zone'].isin([1, 4, 5, 7, 11, 13])]
        metrics['xwOBA_Inside'] = inside['estimated_woba_using_speedangle'].mean()
        
        breaking = batter_df[batter_df['pitch_type'].isin(['SL', 'CB', 'CU', 'KC', 'CS'])]
        metrics['xwOBA_Breaking'] = breaking['estimated_woba_using_speedangle'].mean()
        
        offspeed = batter_df[batter_df['pitch_type'].isin(['CH', 'FS', 'SC', 'FO'])]
        metrics['xwOBA_Offspeed'] = offspeed['estimated_woba_using_speedangle'].mean()
        
        # ========== CATEGORY D: SITUATIONAL ==========
        risp = batter_df[(batter_df['on_2b'].notna()) | (batter_df['on_3b'].notna())]
        metrics['xwOBA_RISP'] = risp['estimated_woba_using_speedangle'].mean()
        
        two_strikes = batter_df[batter_df['strikes'] == 2]
        metrics['xwOBA_2Strikes'] = two_strikes['estimated_woba_using_speedangle'].mean()
        
        batter_hand = batter_df['stand'].iloc[0] if len(batter_df) > 0 else 'R'
        same_side = batter_df[batter_df['p_throws'] == batter_hand]
        metrics['xwOBA_SameSide'] = same_side['estimated_woba_using_speedangle'].mean()
        
        metrics['xwOBA_RepeatPitch'] = batter_df[batter_df['pitch_number'] > 1]['estimated_woba_using_speedangle'].mean()
        
        hitters_counts = batter_df[
            ((batter_df['balls'] == 2) & (batter_df['strikes'] == 0)) |
            ((batter_df['balls'] == 3) & (batter_df['strikes'] == 1)) |
            ((batter_df['balls'] == 3) & (batter_df['strikes'] == 0))
        ]
        metrics['xwOBA_HittersCount'] = hitters_counts['estimated_woba_using_speedangle'].mean()
        
        return metrics


# ========== CREATE BATTER INSTANCES ==========
metrics_engine = WeaknessMetrics(df)
all_batters = []

for i, mlbam_id in enumerate(mlbam_ids):
    if mlbam_id is None:
        continue
    
    name = batter_names[i]
    
    # Check if batter is a switch hitter
    batter_df = df[df['batter'] == mlbam_id]
    if len(batter_df) == 0:
        continue
    
    handedness_values = batter_df['stand'].unique()
    
    # If switch hitter, create two instances (one for each side)
    if len(handedness_values) > 1 or 'S' in handedness_values:
        print(f"Processing switch hitter {name}")
        
        # Left-handed instance
        metrics_L = metrics_engine.calculate_all(mlbam_id, handedness='L')
        if metrics_L:
            batter_L = Batter(mlbam_id, name, 'L')
            batter_L.add_metrics(metrics_L)
            all_batters.append(batter_L)
        
        # Right-handed instance
        metrics_R = metrics_engine.calculate_all(mlbam_id, handedness='R')
        if metrics_R:
            batter_R = Batter(mlbam_id, name, 'R')
            batter_R.add_metrics(metrics_R)
            all_batters.append(batter_R)
    else:
        # Regular batter (single handedness)
        handedness = handedness_values[0]
        print(f"Processing batter {len(all_batters) + 1}: {name} ({handedness})")
        
        metrics = metrics_engine.calculate_all(mlbam_id)
        if metrics:
            batter = Batter(mlbam_id, name, handedness)
            batter.add_metrics(metrics)
            all_batters.append(batter)


# ========== CALCULATE PERCENTILES ==========
# Collect all metrics into a dataframe for percentile calculation
metrics_data = []
for batter in all_batters:
    row = {'batter_id': batter.batter_id, 'name': batter.name, 'handedness': batter.handedness}
    row.update(batter.metrics)
    metrics_data.append(row)

metrics_df = pd.DataFrame(metrics_data)

# Define which metrics are "low = good" vs "low = bad"
low_is_bad = [
    'xwOBACON', 'HardHit%', 'Barrel%', 'AvgEV', 'SweetSpot%',
    'ZoneContact%', 'xwOBA_HighFB', 'xwOBA_LowAway', 'xwOBA_Inside',
    'xwOBA_Breaking', 'xwOBA_Offspeed', 'xwOBA_RISP', 'xwOBA_2Strikes',
    'xwOBA_SameSide', 'xwOBA_RepeatPitch', 'xwOBA_HittersCount'
]

high_is_bad = [
    'Chase%', 'Whiff%', 'FirstPitchSwing%', 'CalledStrike%'
]

# percentiles
percentile_data = {}
for metric in low_is_bad:
    if metric in metrics_df.columns:
        percentile_data[metric] = metrics_df[metric].rank(pct=True) * 100

for metric in high_is_bad:
    if metric in metrics_df.columns:
        percentile_data[metric] = (100 - (metrics_df[metric].rank(pct=True) * 100))

for idx, batter in enumerate(all_batters):
    batter_percentiles = {metric: percentile_data[metric].iloc[idx] for metric in percentile_data}
    batter.add_percentiles(batter_percentiles)
    batter.identify_weaknesses(top_n=3)
    batter.identify_strengths(top_n=3)


# ========== GENERATE OUTPUTS ==========
profiles = []
for batter in all_batters:
    if len(batter.weaknesses) > 0:
        profile = batter.generate_profile()
        profiles.append(profile)

# Print first profile as example
if profiles:
    print(profiles[0])

# Save all profiles to file
with open('batter_weakness_profiles.txt', 'w', encoding='utf-8') as f:
    for profile in profiles:
        f.write(profile + "\n\n")

# Save raw metrics to CSV
metrics_df.to_csv('batter_metrics_complete.csv', index=False)
print(f"\nSaved {len(all_batters)} batter instances to files")
print("Saved: batter_metrics_complete.csv")
print("Saved: batter_weakness_profiles.txt")