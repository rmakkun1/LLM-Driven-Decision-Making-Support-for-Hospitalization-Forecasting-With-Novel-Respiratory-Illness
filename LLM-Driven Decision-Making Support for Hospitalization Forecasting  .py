"""
COVID-19 Hospitalization Prediction Analysis
Compares LLM predictions with actual hospitalization data for Pennsylvania counties.
Uses April 2021 data to predict May 2021 hospitalizations.

Author: Rhea Makkuni
Date: 2025
"""

import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
import seaborn as sns
import random
import time
import math
import unicodedata
import requests
from collections import Counter, defaultdict
import re
import io
import openai
from openai import OpenAI
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# OpenAI client setup
client = OpenAI(api_key="API KEY HERE")

def query_llm_simple(prompt, model="gpt-4"):
    """Simple function to query LLM"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None
    
# Configuration
PI_LABS_API_KEY = "API KEY HERE"

# Focus on counties and specific time period
counties = ['Bradford', 'Forest', 'Allegheny', 'Philadelphia', 'Montgomery', 'Bucks']
geographies = counties  
training_month = 'April 2021'
prediction_month = 'May 2021'
iterations_per_county = 1

def load_covid_data(file_path=None):
    """
    Load COVID-19 hospitalization data from CSV file.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the CSV file. If None, uses default Downloads path.
    
    Returns:
    --------
    pandas.DataFrame or None
        Loaded dataframe or None if loading fails
    """
    
    # Default file path if none provided
    if file_path is None:
        file_path = '/Users/rheamakkuni/Downloads/COVID-19_Aggregate_Hospitalizations_Current_Weekly_County_Health_20251019.csv'
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Loaded hospitalization data: {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        if 'County' in df.columns:
            unique_counties = df['County'].dropna().unique()
            print(f"Available counties (first 20): {list(unique_counties[:20])}")
            print(f"Total unique counties: {len(unique_counties)}")
        
        return df
        
    except FileNotFoundError:
        print("CSV file not found. Please ensure the file exists in the specified directory.")
        return None
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

class PiLabsClient:
    """Pi Labs REST API client"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.withpi.ai"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.scoring_system = self
    
    def score(self, llm_input, llm_output, scoring_spec):
        """Score method for compatibility"""
        try:
            payload = {
                "llm_input": llm_input,
                "llm_output": llm_output,
                "scoring_spec": scoring_spec
            }
            
            response = requests.post(
                f"{self.base_url}/v1/score",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                score_value = result.get('total_score') or result.get('score') or result.get('result')
                return MockScore(float(score_value) if score_value is not None else np.nan)
            else:
                return MockScore(np.nan)
                
        except Exception as e:
            return MockScore(np.nan)

class MockScore:
    def __init__(self, score):
        self.total_score = score

def initialize_pi_client():
    """Initialize Pi Labs client"""
    try:
        if PI_LABS_API_KEY == "API KEY HERE":
            print("Please set your Pi Labs API key!")
            return None
        
        pi_client = PiLabsClient(PI_LABS_API_KEY)
        print("Pi Labs REST client initialized")
        return pi_client
    except Exception as e:
        print(f"Failed to initialize Pi Labs client: {e}")
        return None

def random_month_generator():
    return training_month

# Pi Labs extraction functions
def pi_labs_extract_number(llm_response: str, pi_client) -> float:
    if pi_client is None:
        return np.nan
    try:
        scoring_spec = {"name": "hospitalization_rate_percent", "type": "number", "min": 0, "max": 100, "units": "percent"}
        score_obj = pi_client.scoring_system.score(llm_input="", llm_output=llm_response, scoring_spec=scoring_spec)
        value = getattr(score_obj, "total_score", np.nan)
        if value is None or np.isnan(value):
            return np.nan
        return float(value) if 0 <= float(value) <= 100 else np.nan
    except Exception:
        return np.nan

def _normalize(text: str) -> str:
    t = unicodedata.normalize('NFKC', text)
    t = re.sub(r'(?i)covid\D*19', 'COVID', t)
    return t

POSITIVE_KEYWORDS = (
    "hospitalization", "hospitalisation", "hospitalized", "hospitalised",
    "hospital", "admission", "admissions", "inpatient", "icu", "ward",
    "bed", "rate", "rates"
)
NEGATIVE_KEYWORDS = (
    "positivity", "test positivity", "positive rate", "cases", "case",
    "incidence", "infection", "vaccine", "vaccination", "efficacy",
    "mortality", "death", "fatality"
)
MONTH_WORDS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec")

LABEL_PATTERNS = [
    r'(hospitali\w+|admission|inpatient|icu|rate|rates)[^%.]{0,40}(?P<val>\d{1,3}(?:\.\d+)?)\s?%',
    r'(?P<val>\d{1,3}(?:\.\d+)?)\s?%\s*(hospitali\w+|admission|inpatient|icu|rate|rates)',
    r'(hospitali\w+|admission|inpatient|icu|rate|rates)[^%.]{0,40}(?P<val>\d{1,3}(?:\.\d+)?)\s*(?:percent|per\s*cent)',
    r'(?P<val>\d{1,3}(?:\.\d+)?)\s*(?:percent|per\s*cent)\s*(hospitali\w+|admission|inpatient|icu|rate|rates)',
]

def _split_sentences(text: str):
    parts = re.split(r'(?<=[\.\?\!;])\s+', text)
    offsets = []
    pos = 0
    for p in parts:
        start = text.find(p, pos)
        if start == -1:
            start = pos
        offsets.append((p, start, start + len(p)))
        pos = start + len(p)
    return offsets

def _candidate_sentence_idx(span_start: int, sentences):
    for idx, (_s, a, b) in enumerate(sentences):
        if a <= span_start < b:
            return idx
    return -1

def _contains_obvious_year(span_start: int, span_end: int, text: str) -> bool:
    left = max(0, span_start - 10)
    right = min(len(text), span_end + 10)
    ctx = text[left:right]
    if re.search(r'\b(19|20)\d{2}\b', ctx) and re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b', ctx, re.IGNORECASE):
        return True
    return False

def _extract_percent_candidates(text: str):
    cands = []
    for pat in LABEL_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            try:
                v = float(m.group('val'))
            except Exception:
                continue
            if 0 <= v <= 100:
                cands.append((v, m.start('val'), m.end('val'), True))
    generic_patterns = [
        r'(?P<val>\d{1,3}(?:\.\d+)?)\s*%',
        r'(?P<val>\d{1,3}(?:\.\d+)?)\s*(?:per\s*cent|percent)'
    ]
    for pat in generic_patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            try:
                v = float(m.group('val'))
            except Exception:
                continue
            if 0 <= v <= 100:
                key = (m.start('val'), m.end('val'))
                if not any(s == key[0] and e == key[1] for _, s, e, _ in cands):
                    cands.append((v, m.start('val'), m.end('val'), False))
    return cands

def _near_any(words, window: str, left_offset: int, center: int, weight: float):
    score = 0.0
    for kw in words:
        for m in re.finditer(re.escape(kw), window, flags=re.IGNORECASE):
            dist = abs((left_offset + m.start()) - center) + 1
            score += weight / dist
    return score

def _has_near(text: str, target_center: int, words, max_dist: int) -> bool:
    for kw in words:
        for m in re.finditer(re.escape(kw), text, flags=re.IGNORECASE):
            if abs(m.start() - target_center) <= max_dist or abs(m.end() - target_center) <= max_dist:
                return True
    return False

def _score_candidate(val: float, span_start: int, span_end: int, label_tied: bool, text: str, sentences) -> float:
    center = (span_start + span_end) // 2
    win_r = 140
    left = max(0, center - win_r)
    right = min(len(text), center + win_r)
    window = text[left:right]

    sidx = _candidate_sentence_idx(span_start, sentences)
    sent_text, _sa, _sb = sentences[sidx] if sidx != -1 else ("", 0, 0)

    score = 0.0
    if label_tied:
        score += 600.0

    score += _near_any(POSITIVE_KEYWORDS, window, left, center, weight=300.0)
    score -= _near_any(NEGATIVE_KEYWORDS, window, left, center, weight=320.0)

    if _has_near(text, center, POSITIVE_KEYWORDS, max_dist=35):
        score += 250.0
    if _has_near(text, center, NEGATIVE_KEYWORDS, max_dist=35) and not _has_near(text, center, POSITIVE_KEYWORDS, max_dist=35):
        score -= 300.0

    if re.search(r'|'.join(re.escape(k) for k in NEGATIVE_KEYWORDS), sent_text, flags=re.IGNORECASE) and not re.search(r'|'.join(re.escape(k) for k in POSITIVE_KEYWORDS), sent_text, flags=re.IGNORECASE):
        score -= 800.0

    month_hit = any(re.search(rf'\b{m}[a-z]*\b', sent_text, flags=re.IGNORECASE) for m in MONTH_WORDS)
    if month_hit and not re.search(r'|'.join(re.escape(k) for k in POSITIVE_KEYWORDS), sent_text, flags=re.IGNORECASE):
        score -= 300.0

    if re.search(r'%\s*$', text[span_start:span_end]):
        score += 2.0

    return score

def _has_any_percent_over_100(text: str) -> bool:
    if re.search(r'\b(1\d{2}|\d{3,})\s*%', text):
        return True
    if re.search(r'\b(1\d{2}|\d{3,})\s*(?:percent|per\s*cent)\b', text, flags=re.IGNORECASE):
        return True
    return False

def _has_any_label_tied_leq_100(text: str) -> bool:
    for pat in LABEL_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            try:
                v = float(m.group('val'))
            except Exception:
                continue
            if 0 <= v <= 100:
                return True
    return False

def pi_labs_extract_with_fallback_improved(llm_response, pi_client):
    text_raw = llm_response if isinstance(llm_response, str) else str(llm_response)
    text = _normalize(text_raw)

    # First try to extract simple numbers (not percentages)
    try:
        # Look for simple numbers in the response
        number_patterns = [
            r'\b(\d{1,6})\b',  # Simple numbers like 7650, 37, 298
            r'(\d{1,6})',       # Any number
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Take the largest number found (most likely to be the prediction)
                numbers = [int(match) for match in matches if match.isdigit()]
                if numbers:
                    largest_number = max(numbers)
                    # Accept any reasonable hospitalization number (0 to 100,000)
                    if 0 <= largest_number <= 100000:
                        return float(largest_number), "simple_number"
    except Exception:
        pass

    # Fallback to original Pi Labs extraction
    try:
        pi_val = pi_labs_extract_number(text, pi_client)
        if pi_val is not None and not np.isnan(pi_val) and 0 <= pi_val <= 100:
            return float(pi_val), "pi_labs"
    except Exception:
        pass

    if _has_any_percent_over_100(text) and not _has_any_label_tied_leq_100(text):
        return np.nan, "failed"

    try:
        cands = _extract_percent_candidates(text)
        if not cands:
            return np.nan, "failed"

        cands = [(v, s, e, tied) for (v, s, e, tied) in cands if not _contains_obvious_year(s, e, text)]
        if not cands:
            return np.nan, "failed"

        sentences = _split_sentences(text)

        invalid_sent_idxs = set()
        for idx, (sent_text, sa, sb) in enumerate(sentences):
            if re.search(r'\binvalid\b|not\s+valid|typo|misreport', sent_text, flags=re.IGNORECASE) and \
               re.search(r'\b(1\d{2}|\d{3,})\s*(?:%|percent|per\s*cent)\b', sent_text, flags=re.IGNORECASE):
                invalid_sent_idxs.add(idx)

        def in_invalid_sentence(span_start: int) -> bool:
            si = _candidate_sentence_idx(span_start, sentences)
            return si in invalid_sent_idxs

        if invalid_sent_idxs:
            filtered = [(v, s, e, tied) for (v, s, e, tied) in cands if not in_invalid_sentence(s)]
            if not filtered and not any(tied and 0 <= v <= 100 for (v, _s, _e, tied) in cands if not in_invalid_sentence(_s)):
                return np.nan, "failed"
            cands = filtered or cands

        if not cands:
            return np.nan, "failed"

        scored = [(v, _score_candidate(v, s, e, tied, text, sentences)) for (v, s, e, tied) in cands]
        best_val, best_score = max(scored, key=lambda x: x[1])

        if best_score < -20.0:
            return np.nan, "failed"

        return float(best_val), "fallback" if 0 <= best_val <= 100 else (np.nan, "failed")
    except Exception:
        return np.nan, "failed"

pi_labs_extract_with_fallback = pi_labs_extract_with_fallback_improved

# Data formatting and prediction functions
def format_data_for_llm(df, target_month, geography, lookback_weeks=4):
    """Format recent hospitalization data for LLM consumption"""
    
    if df is None:
        return "No data available - CSV file not loaded."
    
    try:
        # Convert month string to date for filtering
        month_name, year = target_month.split()
        month_num = datetime.strptime(month_name[:3], "%b").month
        target_date = pd.to_datetime(f"{year}-{month_num:02d}-01")
        start_date = target_date - pd.Timedelta(weeks=lookback_weeks)
        
        # Convert date column if needed
        if 'Date of data' in df.columns:
            df['Date of data'] = pd.to_datetime(df['Date of data'], errors='coerce')
        
        # Filter data by geography and date range
        if geography == 'United States':
            recent_data = df[(df['Date of data'] >= start_date) & 
                           (df['Date of data'] < target_date)].copy()
        else:
            # Filter by specific geography - try exact match first, then partial
            geo_filter = (df['County'] == geography) | df['County'].str.contains(geography, case=False, na=False)
            recent_data = df[geo_filter & 
                           (df['Date of data'] >= start_date) & 
                           (df['Date of data'] < target_date)].copy()
        
        if len(recent_data) == 0:
            return f"No recent hospitalization data available for {geography} during {target_month}."
        
        # Debug: Show how much data we found
        print(f"      Found {len(recent_data)} data points for {geography} in {target_month}")
        
        # Format summary
        data_summary = []
        data_summary.append(f"Recent COVID-19 hospitalization data for {geography}:")
        data_summary.append("")
        
        # Get latest available data
        latest_week = recent_data['Date of data'].max()
        latest_data = recent_data[recent_data['Date of data'] == latest_week]
        
        if len(latest_data) > 0:
            # Convert numeric columns to numeric, handling empty strings
            hosp_col = 'COVID-19 Patients Hospitalized'
            avg_col = '14-day average COVID-19 patients hospitalized'
            
            # Convert to numeric, replacing empty strings with NaN
            latest_data = latest_data.copy()
            latest_data[hosp_col] = pd.to_numeric(latest_data[hosp_col], errors='coerce')
            latest_data[avg_col] = pd.to_numeric(latest_data[avg_col], errors='coerce')
            
            # Calculate sums and means, ignoring NaN values
            total_hospitalizations = latest_data[hosp_col].sum()
            avg_14_day = latest_data[avg_col].mean()
            
            data_summary.append(f"Latest week ({latest_week.strftime('%Y-%m-%d')}):")
            data_summary.append(f"  Total COVID-19 hospitalizations: {total_hospitalizations:.0f}")
            data_summary.append(f"  14-day average: {avg_14_day:.1f}")
            
            # Add trend if we have previous data
            prev_weeks = recent_data[recent_data['Date of data'] < latest_week]
            if len(prev_weeks) > 0:
                prev_week = prev_weeks['Date of data'].max()
                prev_data = recent_data[recent_data['Date of data'] == prev_week].copy()
                prev_data[hosp_col] = pd.to_numeric(prev_data[hosp_col], errors='coerce')
                prev_total = prev_data[hosp_col].sum()
                trend = total_hospitalizations - prev_total
                data_summary.append(f"  Weekly change: {trend:+.0f}")
        
        return "\n".join(data_summary)
        
    except Exception as e:
        return f"Error formatting data: {str(e)}"

def data_driven_hospitalization_prompt(profile, recent_data):
    """Create prompt asking LLM to predict May 2021 hospitalizations based on April 2021 data"""
    return f"""Given this April 2021 COVID-19 hospitalization data for {profile['geography']}:

{recent_data}

Based on this April 2021 trend data, predict the total number of COVID-19 hospitalizations for the first week of May 2021 in {profile['geography']}. 

Consider the April trend and give a direct numerical prediction. For example, if April hospitalizations are 2000 and trending up by 100 per week, predict around 2100 for the first week of May.

Give your answer as a single number representing the predicted total hospitalizations for the first week of May 2021."""

def get_actual_may_2021_hospitalizations(df, geography):
    """Get the actual hospitalization numbers for first week of May 2021"""
    
    if df is None:
        return np.nan
    
    try:
        # Get first week of May 2021 (May 1-7, 2021)
        may_start = pd.to_datetime('2021-05-01')
        may_first_week_end = pd.to_datetime('2021-05-07')
        
        # Ensure date column is properly converted
        df_copy = df.copy()
        df_copy['Date of data'] = pd.to_datetime(df_copy['Date of data'], errors='coerce')
        
        # Filter data for first week of May 2021
        if geography == 'United States':
            target_data = df_copy[(df_copy['Date of data'] >= may_start) & 
                                (df_copy['Date of data'] <= may_first_week_end)]
        else:
            geo_filter = df_copy['County'].str.contains(geography.split(',')[0], case=False, na=False)
            target_data = df_copy[geo_filter & 
                                (df_copy['Date of data'] >= may_start) & 
                                (df_copy['Date of data'] <= may_first_week_end)]
        
        if len(target_data) == 0:
            return np.nan
        
        # Convert hospitalization column to numeric, handling empty strings
        hosp_col = 'COVID-19 Patients Hospitalized'
        target_data[hosp_col] = pd.to_numeric(target_data[hosp_col], errors='coerce')
        
        # Calculate average total hospitalizations for the first week of May
        avg_hospitalizations = target_data[hosp_col].mean()
        
        return avg_hospitalizations
        
    except Exception as e:
        return np.nan

def evaluate_response_directness(response):
    """Evaluate if LLM response is direct vs overly questioning"""
    
    question_count = response.count('?')
    has_specific_number = bool(re.search(r'\d+\.?\d*\s*%', response))
    has_prediction_language = any(word in response.lower() for word in 
                                 ['predict', 'forecast', 'expect', 'estimate', 'based on'])
    has_uncertainty_language = any(word in response.lower() for word in 
                                  ['do you mean', 'which', 'should i', 'clarify', 'what type'])
    
    score = 0
    feedback = []
    
    # Positive points for directness
    if has_specific_number:
        score += 3
        feedback.append("Includes specific percentage")
    
    if has_prediction_language:
        score += 2  
        feedback.append("Uses prediction language")
    
    if question_count == 0 and has_specific_number:
        score += 2
        feedback.append("Direct answer without questions")
    
    # Negative points for excessive questioning
    if question_count > 2:
        score -= 3
        feedback.append(f"Too many questions ({question_count})")
    
    if has_uncertainty_language and not has_specific_number:
        score -= 2
        feedback.append("Asks for clarification instead of predicting")
    
    return {
        'directness_score': score,
        'feedback': feedback,
        'classification': 'Direct' if score > 0 else 'Questioning',
        'question_count': question_count,
        'has_number': has_specific_number
    }

def run_hospitalization_prediction_experiment():
    """Data-driven hospitalization prediction experiment"""
    
    print("=" * 60)
    print("COUNTY HOSPITALIZATION PREDICTION EXPERIMENT")
    print("April 2021 data → Predict May 2021 first week → Compare to reality")
    print("=" * 60)
    
    # Load data
    df = load_covid_data()
    
    if df is None:
        print("Cannot run experiment - no data loaded")
        return []
    
    pi_client = initialize_pi_client()
    results = []
    
    # Experiment metadata
    experiment_metadata = {
        'experiment_type': 'county_hospitalization_prediction',
        'timestamp': datetime.now().isoformat(),
        'geographies_tested': geographies,
        'counties_tested': counties,
        'training_month': training_month,
        'prediction_month': prediction_month,
        'iterations_per_county': iterations_per_county
    }
    results.append(experiment_metadata)
    
    # Run experiment for each geography
    for geography in geographies:
        print(f"\nTesting {geography}...")
        
        geo_result = {
            'geography': geography,
            'temporal_data': {}
        }
        
        # Test this geography with multiple iterations
        county_data = {
            'predictions': [],
            'actual_values': [],
            'raw_responses': [],
            'extraction_methods': [],
            'directness_scores': [],
            'iterations': [],
            'accuracy_metrics': {}
        }
        
        # Run multiple iterations for this geography
        for iteration in range(iterations_per_county):
            # Generate profile
            profile = {
                'geography': geography,
                'training_month': training_month,
                'prediction_month': prediction_month
            }
            
            print(f"    Iteration {iteration + 1}/{iterations_per_county}: {training_month} → {prediction_month}")
            print(f"      Testing: {geography}")
            
            # Get April 2021 data for LLM
            recent_data = format_data_for_llm(df, training_month, geography)
            
            # Debug: Show what data is being sent to LLM
            print(f"      Data being sent to LLM:")
            print(f"         {recent_data[:300]}{'...' if len(recent_data) > 300 else ''}")
            
            # Create prediction prompt
            prediction_prompt = data_driven_hospitalization_prompt(profile, recent_data)
            
            try:
                # Get LLM prediction
                llm_response = query_llm_simple(prediction_prompt)
                predicted_number, extraction_method = pi_labs_extract_with_fallback(llm_response, pi_client)
                directness_eval = evaluate_response_directness(llm_response)
                
                # Get actual value for comparison (May 2021 first week)
                actual_value = get_actual_may_2021_hospitalizations(df, geography)
                
                # Show predictions live
                print(f"      LLM Response: {llm_response[:100]}{'...' if len(llm_response) > 100 else ''}")
                if not np.isnan(predicted_number):
                    if not np.isnan(actual_value):
                        error = abs(predicted_number - actual_value)
                        print(f"       Predicted: {predicted_number:.0f} hospitalizations | Actual: {actual_value:.0f} hospitalizations | Error: {error:.0f}")
                    else:
                        print(f"       Predicted: {predicted_number:.0f} hospitalizations | Actual: No data available")
                else:
                    print(f"       Could not extract prediction from response")
                
                # Show directness score
                directness_score = directness_eval['directness_score']
                classification = directness_eval['classification']
                print(f"      Directness: {directness_score}/5 ({classification})")
                print()
                
                # Store results
                county_data['predictions'].append(predicted_number)
                county_data['actual_values'].append(actual_value)
                county_data['raw_responses'].append(llm_response)
                county_data['extraction_methods'].append(extraction_method)
                county_data['directness_scores'].append(directness_eval)
                county_data['iterations'].append(iteration + 1)
                
            except Exception as e:
                print(f"       Prediction failed: {e}")
                county_data['predictions'].append(np.nan)
                county_data['actual_values'].append(np.nan)
                county_data['raw_responses'].append("ERROR")
                county_data['extraction_methods'].append("failed")
                county_data['directness_scores'].append({'directness_score': -1})
                county_data['iterations'].append(iteration + 1)
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Calculate accuracy metrics for this county (after all iterations)
        valid_predictions = [(p, a) for p, a in zip(county_data['predictions'], county_data['actual_values']) 
                           if not np.isnan(p) and not np.isnan(a)]
        
        if valid_predictions:
            predictions, actuals = zip(*valid_predictions)
            mae = np.mean([abs(p - a) for p, a in valid_predictions])
            directness_avg = np.mean([score['directness_score'] for score in county_data['directness_scores']])
            success_rate = len(valid_predictions) / len(county_data['predictions'])
            
            county_data['accuracy_metrics'] = {
                'mean_absolute_error': mae,
                'directness_average': directness_avg,
                'success_rate': success_rate,
                'valid_predictions': len(valid_predictions)
            }
            
            # Show summary for this county
            print(f"     {geography} Summary: {success_rate:.1%} success, {mae:.0f} avg error (hospitalizations), {directness_avg:.1f}/5 directness")