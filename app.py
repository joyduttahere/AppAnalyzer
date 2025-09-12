# BrandManager app.py
# Version 2.12 - Corrected model loading and removed duplicate functions and Gemini specific functions 

import os
import ssl
import re
import time
import torch
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, jsonify
import json
import markdown2
from pyngrok import ngrok
import google.generativeai as genai

# SSL Configuration
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from dotenv import load_dotenv
load_dotenv()

#import nltk
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
from google_play_scraper import reviews, Sort, search
from collections import defaultdict
import requests

# Disable SSL warnings
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig

# ==============================================================================
# 1. CONSTANTS AND CONFIGURATION
# ==============================================================================

PAIN_POINT_CATEGORIES = {
    "App Stability & Performance": {
        "Crashes & Stability": [
            "app crashes", "force close", "force closes", "app closes", "shuts down suddenly",
            "stops working", "won't start", "won't open", "can't open", "fails to launch",
            "black screen", "white screen", "blank screen", "frozen app", "app freezes",
            "hangs up", "hanging", "stuck loading", "unresponsive app", "not responding"
        ],
        "Bugs & Errors": [
            "software bug", "app bug", "buggy behavior", "glitch", "glitches", "glitchy",
            "error message", "error code", "runtime error", "system error", "fatal error",
            "malfunction", "broken feature", "feature broken", "corrupted data", "invalid response",
            "null pointer", "exception error", "404 error", "500 error", "server error"
        ],
        "Performance Issues": [
            "slow loading", "takes forever to load", "loading forever", "endless loading",
            "spinning wheel", "buffering constantly", "app lag", "lagging", "laggy performance",
            "sluggish response", "choppy animation", "stuttering", "frame drops", "delayed response",
            "timeout error", "times out", "slow performance", "performance issues"
        ],
        "Resource Usage": [
            "battery drain", "drains battery fast", "battery consumption", "overheating phone",
            "phone gets hot", "cpu usage high", "memory leak", "ram usage", "storage space",
            "takes up space", "disk space full", "data usage high", "mobile data consumption"
        ]
    },

    "Onboarding & Setup": {
        "Initial Setup Problems": [
            "setup failed", "installation incomplete", "setup wizard broken", "initial configuration error",
            "first time setup", "welcome screen stuck", "setup process confusing", "can't complete setup",
            "setup keeps crashing", "setup won't finish", "configuration failed", "initial setup error",
            "setup tutorial broken", "guided setup issues", "setup verification failed", "setup timeout"
        ],
        "Account Creation Issues": [
            "can't create account", "signup broken", "registration failed", "account creation error",
            "email already exists", "username taken", "password requirements unclear", "terms acceptance failed",
            "captcha not working", "verification email never came", "phone verification failed", "age verification issues",
            "country not supported", "region restrictions", "account creation timeout", "signup form broken"
        ],
        "Tutorial & First Experience": [
            "tutorial skipped accidentally", "tutorial too long", "tutorial confusing", "onboarding unclear",
            "first use confusing", "don't understand how to start", "tutorial broken", "walkthrough issues",
            "intro video won't play", "getting started guide missing", "first steps unclear", "tutorial won't load",
            "skip tutorial button missing", "tutorial navigation broken", "onboarding too complicated", "tutorial crashes"
        ],
        "Profile & Preferences Setup": [
            "profile setup incomplete", "preferences not saving", "avatar upload failed", "profile picture issues",
            "personal information error", "preference settings broken", "profile creation failed", "settings not applying",
            "default preferences wrong", "customization options limited", "profile validation failed", "bio won't save",
            "interest selection broken", "category preferences error", "notification setup failed", "privacy settings unclear"
        ],
        "Permission & Access Setup": [
            "permission denied", "camera access denied", "microphone permission issues", "location permission failed",
            "contacts access blocked", "storage permission required", "notification permission setup", "background app refresh",
            "permission popup keeps appearing", "can't grant permissions", "permission settings confusing", "access request failed",
            "privacy permission unclear", "permission explanation missing", "why does app need permission", "permission too intrusive"
        ],
        "Integration & Import Setup": [
            "can't import contacts", "social media login failed", "google signin broken", "facebook connect issues",
            "import from other app failed", "data migration error", "sync with existing account", "third party integration broken",
            "calendar sync setup failed", "email integration issues", "import csv failed", "backup restore during setup",
            "connect existing service", "link account failed", "oauth authentication error", "api connection timeout"
        ]
    },

    "Connectivity & Integration": {
        "Network & Sync": [
            "connection failed", "can't connect to server", "connection error", "network error",
            "server down", "server unavailable", "no internet connection", "offline issues",
            "sync failed", "sync error", "won't sync", "syncing problems", "cloud sync",
            "api timeout", "network timeout", "connection lost", "unstable connection"
        ],
        "Device Integration": [
            "bluetooth connection", "bluetooth pairing", "bluetooth issues", "camera not working",
            "microphone problems", "speaker issues", "gps not working", "location services",
            "fingerprint scanner", "face id problems", "touch id", "biometric authentication",
            "sensor issues", "accelerometer", "gyroscope problems", "hardware integration"
        ],
        "Platform Compatibility": [
            "not compatible", "incompatible device", "device not supported", "android version",
            "ios version", "operating system", "phone model", "tablet support", "screen resolution",
            "orientation issues", "landscape mode", "portrait mode", "version compatibility"
        ]
    },

    "User Experience": {
        "UI Design & Layout": [
            "ugly interface", "bad design", "poor layout", "cluttered screen", "messy design",
            "confusing layout", "hard to read text", "small text", "tiny buttons", "big buttons",
            "overlapping elements", "misaligned buttons", "cut off text", "weird fonts",
            "color scheme bad", "contrast issues", "dark mode issues", "theme problems"
        ],
        "Navigation & Usability": [
            "hard to navigate", "confusing navigation", "can't find button", "hidden options",
            "menu confusing", "too many steps", "complicated workflow", "not user friendly",
            "not intuitive", "steep learning curve", "hard to use", "difficult interface",
            "where is the setting", "how to use", "instructions unclear", "poor usability"
        ],
        "Accessibility": [
            "accessibility issues", "screen reader", "voice over", "contrast too low",
            "color blind friendly", "font size too small", "vision impaired", "hearing impaired",
            "disability support", "accessible design", "inclusive design", "ada compliance"
        ]
    },

    "Account & Authentication": {
        "Login & Security": [
            "can't login", "login failed", "login error", "password incorrect", "wrong password",
            "forgot password", "reset password", "password reset", "account locked", "locked out",
            "verification failed", "otp not received", "two factor authentication", "2fa issues",
            "authentication error", "signin problems", "signup failed", "email verification"
        ],
        "Account Management": [
            "account deleted", "profile missing", "account suspended", "banned account",
            "account recovery", "lost account", "can't access account", "account settings",
            "profile settings", "privacy settings", "data export", "account deletion"
        ],
        "Privacy & Security": [
            "privacy concerns", "data collection", "personal information", "location tracking",
            "data sharing", "third party access", "privacy policy", "data breach",
            "security concern", "not secure", "unsafe app", "suspicious activity",
            "account hacked", "unauthorized access", "identity theft", "data stolen"
        ]
    },

    "Business & Service": {
        "Customer Support": [
            "customer support", "customer service", "support team", "help desk", "contact support",
            "no response from support", "poor customer service", "rude support staff", "unhelpful support",
            "support chat", "support email", "support ticket", "complaint handling", "escalation needed",
            "supervisor", "manager", "support quality", "response time", "resolution time"
        ],
        "Billing & Payments": [
            "charged twice", "double charged", "billing error", "payment failed", "can't make payment",
            "payment declined", "credit card declined", "refund request", "money back", "overcharged",
            "billing issue", "subscription problem", "auto renewal", "cancel subscription", "billing cycle",
            "payment method", "paypal issues", "credit card problem", "transaction failed"
        ],
        "Product Quality": [
            "poor quality product", "low quality", "cheap quality", "defective item", "damaged product",
            "wrong item sent", "not as described", "misleading description", "fake product", "counterfeit",
            "not authentic", "not genuine", "not original", "brand quality", "quality control"
        ],
        "Delivery & Fulfillment": [
            "late delivery", "delayed shipping", "never arrived", "lost package", "missing order",
            "wrong address", "delivery issues", "courier problems", "tracking problems", "no tracking",
            "shipping cost", "delivery charges", "fast delivery", "same day delivery",
            "express shipping", "standard delivery", "out for delivery", "delivery notification"
        ]
    },

    "Features & Functionality": {
        "Missing Features": [
            "need this feature", "missing feature", "add this feature", "feature request", "should have",
            "wish it had", "would be nice", "suggestion for improvement", "enhancement request",
            "new feature needed", "update needed", "improvement needed", "missing functionality"
        ],
        "Broken Features": [
            "feature not working", "feature broken", "stopped working", "removed feature", "disabled feature",
            "limited functionality", "restricted access", "paywall feature", "premium only", "subscription required",
            "feature behind paywall", "locked feature", "unavailable feature", "feature missing"
        ],
        "Search & Discovery": [
            "search not working", "can't find items", "search results", "filter not working", "sort options",
            "browse categories", "recommendation engine", "suggested items", "algorithm issues",
            "no search results", "search function", "discovery features", "find products"
        ]
    },

    "Content & Information": {
        "Content Quality": [
            "poor content", "low quality content", "inaccurate information", "wrong information", "outdated content",
            "content not updated", "fresh content", "relevant content", "useful content", "useless content",
            "spam content", "inappropriate content", "offensive content", "content moderation"
        ],
        "Information Management": [
            "data backup", "data restore", "export data", "import data", "sync data", "cloud storage",
            "save progress", "lost data", "deleted information", "missing data", "recover data",
            "data recovery", "backup failed", "restore failed", "data corruption"
        ]
    },

    "Monetization & Advertising": {
        "Pricing & Subscriptions": [
            "too expensive", "overpriced", "price increase", "subscription cost", "premium price",
            "free version limited", "trial period", "subscription management", "pricing plans",
            "discount codes", "promotional offers", "payment plans", "cost comparison"
        ],
        "Advertisements": [
            "too many ads", "annoying ads", "popup ads", "video ads", "banner ads", "intrusive ads",
            "ad frequency", "ad blocker", "remove ads", "ad free version", "sponsored content",
            "advertising policy", "relevant ads", "targeted ads", "ad personalization"
        ],
        "In-App Purchases": [
            "in app purchase", "microtransaction", "buy credits", "purchase coins", "unlock features",
            "premium upgrade", "purchase failed", "receipt issues", "restore purchase", "refund purchase",
            "iap problems", "store issues", "payment processing", "purchase verification"
        ]
    }
}

# Critical issues that require immediate attention
CRITICAL_ISSUES = [
    # Security & Privacy
    "account hacked", "security breach", "data stolen", "identity theft", "credit card stolen",
    "money stolen", "unauthorized charges", "fraudulent activity", "scam", "phishing attempt",
    "malware detected", "virus", "suspicious activity", "data breach", "privacy violation",

    # Financial
    "charged without permission", "double billing", "can't cancel subscription", "unauthorized transaction",
    "payment error", "billing fraud", "money disappeared", "refund denied", "overcharged significantly",

    # Data Loss
    "lost all data", "data deleted permanently", "can't recover data", "backup failed completely",
    "account deleted", "profile disappeared", "history lost", "progress lost", "work lost",

    # Critical Functionality
    "emergency feature broken", "safety issue", "medical emergency", "urgent problem", "life threatening",
    "dangerous malfunction", "harmful content", "child safety", "inappropriate content for kids",

    # Critical Onboarding Issues
    "can't create account at all", "completely locked out during setup", "setup crashes repeatedly",
    "account creation fraud detected", "setup security warning", "onboarding data breach"
]

# Positive sentiment indicators
POSITIVE_INDICATORS = [
    "love this app", "great app", "awesome feature", "excellent service", "amazing experience",
    "fantastic update", "wonderful design", "perfect functionality", "best app ever", "outstanding quality",
    "brilliant idea", "superb performance", "impressed with", "highly recommend", "five stars",
    "thank you", "grateful for", "appreciate the", "very helpful", "extremely useful",
    "user friendly", "easy to use", "fast loading", "quick response", "smooth experience",
    "reliable app", "stable performance", "solid app", "works perfectly", "no issues",
    "smooth onboarding", "easy setup", "great first impression", "intuitive setup", "seamless registration"
]

# Negative sentiment indicators
NEGATIVE_INDICATORS = [
    "hate this app", "terrible experience", "awful service", "horrible design", "worst app ever",
    "completely useless", "total garbage", "absolute trash", "extremely disappointed", "very frustrating",
    "incredibly annoying", "really irritating", "makes me angry", "absolutely furious", "unacceptable quality",
    "disgusting behavior", "pathetic service", "ridiculous problems", "waste of time", "waste of money",
    "regret downloading", "sorry I installed", "big mistake", "never again", "avoid this app",
    "terrible onboarding", "confusing setup", "horrible first experience", "setup nightmare", "onboarding disaster"
]

# Enhanced feature request detection patterns
FEATURE_REQUEST_INDICATORS = [
    # Direct requests
    "need this feature", "add this feature", "feature request", "should have",
    "wish it had", "would be nice", "suggestion for improvement", "enhancement request",
    "new feature needed", "update needed", "improvement needed", "missing functionality",
    "would be great", "it would be nice", "please add", "can you add", "hope you add",
    "looking forward to", "waiting for", "expecting", "anticipating",
    
    # Comparative requests
    "other apps have", "competitor has", "like in other apps", "similar to", "compared to",
    "as good as", "better than", "improve upon", "upgrade from",
    
    # Conditional/wishful language
    "if you could", "would love", "would appreciate", "hoping for", "wish you would",
    "it should", "could use", "needs to", "ought to", "might want to",
    "consider adding", "think about", "what about", "how about",
    
    # Specific missing features (from your existing list)
    "widget option", "calendar grid layout", "birthdays with year", "important dates marked",
    "holidays marked", "festival dates", "shrink down", "month view", "year view"
]

# Pre-compile critical issues for faster matching
CRITICAL_ISSUES_SET = set(CRITICAL_ISSUES)
POSITIVE_INDICATORS_SET = set(POSITIVE_INDICATORS)
NEGATIVE_INDICATORS_SET = set(NEGATIVE_INDICATORS)

# Use PAIN_POINT_CATEGORIES as TOPIC_CATEGORIES for consistency
TOPIC_CATEGORIES = PAIN_POINT_CATEGORIES
# ==============================================================================
# 2. CORRECTED MODEL LOADING
# ==============================================================================

# Global cache for models
MODEL_CACHE = {}
# Global cache for reviews
REVIEW_CACHE = {}
CACHE_KEY = None  # Current cache identifier
# Global cache for AI-generated results to reduce API calls
AI_RESULT_CACHE = {}

def get_models(selected_model=None):
    """Initializes and returns the AI models with corrected caching and model selection."""
    global MODEL_CACHE
    
    # Handle Gemini API selection
    if selected_model is None or selected_model == 'Gemini-API':
        print(f"Selected AI Model: Gemini-API (cloud-based)")
        # For Gemini API, we still need the classifier model for sentiment analysis
        if "classifier" not in MODEL_CACHE:
            print("Loading Classifier Model for Gemini-API mode...")
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                classifier_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
                classifier_model = AutoModelForSequenceClassification.from_pretrained(
                    "cardiffnlp/twitter-roberta-base-sentiment-latest"
                ).to(DEVICE)
                print("Classifier model loaded.")
                
                MODEL_CACHE["classifier"] = {
                    'tokenizer': classifier_tokenizer,
                    'model': classifier_model
                }
            except Exception as e:
                print(f"Error loading classifier model: {e}")
                raise
        
        return {
            'device': "cuda" if torch.cuda.is_available() else "cpu",
            'classifier_tokenizer': MODEL_CACHE["classifier"]['tokenizer'],
            'classifier_model': MODEL_CACHE["classifier"]['model'],
            'reasoning_tokenizer': None,  # Not needed for Gemini API
            'reasoning_model': None,      # Not needed for Gemini API
            'selected_model': 'Gemini-API'
        }
    
    # For local models, set default if none specified
    if selected_model is None:
        selected_model = "microsoft/DialoGPT-medium"
    
    print(f"Selected AI Model: {selected_model} (local)")
    
    cache_key = f"reasoning_{selected_model}"
    
    # Check if both models are already cached
    if "classifier" in MODEL_CACHE and cache_key in MODEL_CACHE:
        print(f"Models already loaded from cache: classifier + {selected_model}")
        return {
            'device': "cuda" if torch.cuda.is_available() else "cpu",
            'classifier_tokenizer': MODEL_CACHE["classifier"]['tokenizer'],
            'classifier_model': MODEL_CACHE["classifier"]['model'],
            'reasoning_tokenizer': MODEL_CACHE[cache_key]['tokenizer'],
            'reasoning_model': MODEL_CACHE[cache_key]['model'],
            'selected_model': MODEL_CACHE[cache_key]['model_name']
        }
    
    print(f"Loading models: classifier + {selected_model}")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # Load classifier model if not cached
    if "classifier" not in MODEL_CACHE:
        print("Loading Classifier Model...")
        try:
            classifier_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
            classifier_model = AutoModelForSequenceClassification.from_pretrained(
                "cardiffnlp/twitter-roberta-base-sentiment-latest"
            ).to(DEVICE)
            print("Classifier model loaded.")
              
            MODEL_CACHE["classifier"] = {
                'tokenizer': classifier_tokenizer,
                'model': classifier_model
            }
        except Exception as e:
            print(f"Error loading classifier model: {e}")
            raise
        
    # Load reasoning model if not cached
    if cache_key not in MODEL_CACHE:
        print(f"Loading Reasoning Model: {selected_model}...")
          
        try:
            # Unified Loading Logic for different model types
            if any(name in selected_model for name in ["mistral", "SmolLM", "phi-2"]):
                try:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True, 
                        bnb_4bit_quant_type="nf4", 
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    reasoning_model = AutoModelForCausalLM.from_pretrained(
                        selected_model,
                        quantization_config=quant_config,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    print(f"Loaded {selected_model} with 4-bit quantization.")
                except Exception as e:
                    print(f"4-bit quantization failed for {selected_model}: {e}. Falling back to float16.")
                    reasoning_model = AutoModelForCausalLM.from_pretrained(
                        selected_model,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
            else:  # For smaller models like distilgpt2, dialogpt, etc.
                reasoning_model = AutoModelForCausalLM.from_pretrained(
                    selected_model,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(DEVICE)
            
            reasoning_tokenizer = AutoTokenizer.from_pretrained(selected_model, trust_remote_code=True)
            if reasoning_tokenizer.pad_token is None:
                reasoning_tokenizer.pad_token = reasoning_tokenizer.eos_token

            MODEL_CACHE[cache_key] = {
                'tokenizer': reasoning_tokenizer,
                'model': reasoning_model,
                'model_name': selected_model
            }
            print(f"Reasoning model {selected_model} loaded and cached.")
            
        except Exception as e:
            print(f"CRITICAL ERROR loading reasoning model {selected_model}: {e}")
            raise

    print("Model loading complete.")
    return {
        'device': DEVICE,
        'classifier_tokenizer': MODEL_CACHE["classifier"]['tokenizer'],
        'classifier_model': MODEL_CACHE["classifier"]['model'],
        'reasoning_tokenizer': MODEL_CACHE[cache_key]['tokenizer'],
        'reasoning_model': MODEL_CACHE[cache_key]['model'],
        'selected_model': MODEL_CACHE[cache_key]['model_name']
    }

# Flask app initialization
app = Flask(__name__)

# Safely set environment variables only if they have values
ngrok_authtoken = os.environ.get('NGROK_AUTHTOKEN')
hf_token = os.environ.get('HF_TOKEN')
gemini_key = os.environ.get('GEMINI_API_KEY')

# Only set environment variables if they're not None
if ngrok_authtoken:
    os.environ['NGROK_AUTHTOKEN'] = ngrok_authtoken
if hf_token:
    os.environ['HF_TOKEN'] = hf_token
if gemini_key:
    os.environ['GEMINI_API_KEY'] = gemini_key

# Verify they're set
print("Environment variables status:")
print(f"NGROK_AUTHTOKEN: {'✓' if ngrok_authtoken else '✗ (not found)'}")
print(f"HF_TOKEN: {'✓' if hf_token else '✗ (not found)'}")  
print(f"GEMINI_API_KEY: {'✓' if gemini_key else '✗ (not found)'}")

# ==============================================================================
# 3. UTILITY FUNCTIONS
# ==============================================================================

def force_close_connection(func):
    """Decorator to force close HTTP connections."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        original_init = requests.Session.__init__
        
        def new_init(self, *a, **k):
            original_init(self, *a, **k)
            self.headers['Connection'] = 'close'
        
        requests.Session.__init__ = new_init
        try:
            result = func(*args, **kwargs)
        finally:
            requests.Session.__init__ = original_init
        return result
    return wrapper

def clear_gpu_cache():
    """Enhanced GPU cache clearing with better memory management."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Optional: Print memory usage after cleanup
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory after cleanup - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")

def format_installs(install_str):
    """Format installation count strings."""
    if not install_str:
        return "N/A"
    
    try:
        num = int(install_str.replace(',', '').replace('+', ''))
        if num >= 1_000_000_000:
            return f"{num // 1_000_000_000}B+"
        elif num >= 1_000_000:
            return f"{num // 1_000_000}M+"
        elif num >= 1_000:
            return f"{num // 1_000}K+"
        return str(num)
    except (ValueError, TypeError):
        return install_str

def parse_date_range_string(date_str):
    """Parse date range string into datetime objects."""
    if not date_str or ' - ' not in date_str:
        return None, None
    
    try:
        start_str, end_str = date_str.split(' - ')
        start_date = datetime.strptime(start_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_str, "%Y-%m-%d")
        return start_date, end_date
    except (ValueError, TypeError):
        return None, None

def generate_cache_key(app_id, date_range_1_str, date_range_2_str):
    """Generate a unique cache key for app_id and date ranges."""
    return f"{app_id}_{date_range_1_str}_{date_range_2_str}"

def clear_review_cache():
    """Clear the review cache when parameters change."""
    global REVIEW_CACHE, CACHE_KEY
    REVIEW_CACHE.clear()
    CACHE_KEY = None
    print("Review cache cleared due to parameter change.")

def truncate_review_content(content, max_chars=250):
    """Truncate review content to specified character limit."""
    if len(content) <= max_chars:
        return content
    return content[:max_chars - 3] + "..."

# ==============================================================================
# 4. CORE ANALYSIS FUNCTIONS
# ==============================================================================

def identify_feature_requests(review_list):
    """Enhanced feature request identification with better patterns."""
    feature_requests = []
    
    # Enhanced patterns for constructive feature requests
    constructive_request_patterns = [
        # Positive suggestion patterns
        "would be great if", "would love to see", "would be nice to have", 
        "suggestion:", "feature request:", "please consider adding",
        "hope you add", "looking forward to", "excited to see",
        "would appreciate", "would help if you added",
        
        # Constructive improvement language
        "could be improved by", "would be better with", "enhance by adding",
        "consider implementing", "what about adding", "how about",
        "might want to add", "think about adding", "idea for improvement",
        
        # Comparative constructive requests  
        "other apps have this feature", "like app X has", "similar to",
        "would make it as good as", "to compete with", "industry standard",
        
        # Future-oriented positive language
        "in future updates", "next version could include", "roadmap should include",
        "development should focus on", "upcoming features",
        
        # Polite request indicators
        "kindly add", "please include", "would be wonderful", "amazing if you added",
        "dream feature would be", "wishlist item", "feature wishlist"
    ]
    
    # Patterns that indicate complaints rather than requests (to exclude)
    complaint_patterns = [
        "missing feature is annoying", "lack of feature is frustrating",
        "why don't you have", "should already have", "basic feature missing",
        "can't believe there's no", "ridiculous that you don't have",
        "every app has this except", "behind the times", "outdated app"
    ]
    
    for review in review_list:
        content_lower = review['content'].lower()
        
        # FIRST: Exclude reviews with negative sentiment (< -0.1 threshold)
        if review['sentiment_score'] < -0.1:
            continue
            
        # SECOND: Exclude reviews that contain complaint language
        has_complaint_language = any(pattern in content_lower for pattern in complaint_patterns)
        if has_complaint_language:
            continue
            
        # THIRD: Check for constructive feature request patterns
        has_constructive_request = any(pattern in content_lower for pattern in constructive_request_patterns)
        
        # FOURTH: Check for missing features mentioned constructively
        has_constructive_missing_feature = any(
            keyword in content_lower 
            for keyword in PAIN_POINT_CATEGORIES["Features & Functionality"]["Missing Features"]
        )
        
        if has_constructive_request or has_constructive_missing_feature:
            # Determine urgency based on positive language intensity and user sentiment
            urgency = 'low'  # Default to low for feature requests
            
            # Higher urgency for highly positive users making requests
            if review['sentiment_score'] > 0.5:
                if any(urgent_word in content_lower for urgent_word in 
                      ['really need', 'would be amazing', 'game changer', 'must have for me']):
                    urgency = 'medium'
                else:
                    urgency = 'low'
            elif review['sentiment_score'] > 0.2:
                urgency = 'low'
            
            feature_requests.append({
                'review': review,
                'type': 'feature_request',
                'urgency': urgency,
                'content': review['content'],
                'sentiment_score': review['sentiment_score'],
            })
    
    print(f"Identified {len(feature_requests)} constructive feature requests (excluding negative reviews)")
    return feature_requests

def generate_gemini_response(prompt_text):
    """
    CACHING VERSION.
    Sends a prompt to the Gemini API, but first checks a cache to avoid repeated calls.
    """
    global AI_RESULT_CACHE
    # Use the prompt text as the key for the cache
    cache_key = prompt_text

    if cache_key in AI_RESULT_CACHE:
        print(f"✅ Returning cached Gemini response for prompt.")
        return AI_RESULT_CACHE[cache_key]

    try:
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_api_key:
            print("⚠️ Gemini API key not found. Cannot call API.")
            return None

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt_text)

        if not response.parts:
            print("⚠️ Gemini API response was blocked or empty.")
            AI_RESULT_CACHE[cache_key] = None # Cache the failure to avoid retries
            return None

        # Success! Store the result in the cache and return it.
        AI_RESULT_CACHE[cache_key] = response.text
        return response.text

    except Exception as e:
        print(f"❌ An error occurred while calling the Gemini API: {e}")
        AI_RESULT_CACHE[cache_key] = None # Cache the failure
        return None
def generate_feature_themes_gemini(feature_requests):
    """
    GEMINI-POWERED FEATURE THEME GENERATION.
    Asks the Gemini API for a structured JSON response to avoid parsing errors.
    """
    if not feature_requests:
        return {}

    request_texts = "\n".join([f"{i+1}. {req['content']}" for i, req in enumerate(feature_requests[:50])])
    
    # This prompt explicitly asks for a JSON object, which is much more robust to parse.
    prompt = f"""
You are a Senior Product Manager analyzing user feature requests.
Analyze the following requests and group them into 2-4 main themes.

For each theme, provide:
1. A clear "themeName".
2. A 1-2 sentence "summary".
3. A "urgency" rating ('High', 'Medium', or 'Low').
4. The "requestIndices" that belong to this theme.

Respond with ONLY a valid JSON object in a list format, like this:
[
  {{
    "themeName": "Example Theme 1",
    "summary": "Users are asking for...",
    "urgency": "High",
    "requestIndices": [1, 5, 12]
  }},
  {{
    "themeName": "Example Theme 2",
    "summary": "Another group of users wants...",
    "urgency": "Medium",
    "requestIndices": [2, 8]
  }}
]

Here are the user feature requests:
{request_texts}
"""

    response_text = generate_gemini_response(prompt)
    if not response_text:
        return generate_fallback_feature_themes(feature_requests) # Fallback on API error

    try:
        # Find and parse the JSON block from the response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if not json_match:
            print("⚠️ Gemini did not return valid JSON for feature themes. Using fallback.")
            return generate_fallback_feature_themes(feature_requests)
            
        parsed_themes = json.loads(json_match.group(0))
        
        # Convert the parsed JSON into the format your application expects
        final_themes = {}
        for theme in parsed_themes:
            mapped_requests = [feature_requests[i-1] for i in theme.get("requestIndices", []) if 0 < i <= len(feature_requests)]
            final_themes[theme["themeName"]] = {
                'requests': mapped_requests,
                'summary': theme["summary"],
                'urgency': theme["urgency"],
                'count': len(mapped_requests)
            }
        
        print(f"✅ Successfully parsed {len(final_themes)} themes from Gemini JSON.")
        return final_themes

    except Exception as e:
        print(f"Error parsing Gemini JSON for feature themes: {e}. Using rule-based fallback.")
        return generate_fallback_feature_themes(feature_requests)


def generate_feature_themes_with_llm(feature_requests, selected_model=None):
    """Generate feature themes using either Gemini API or local LLM."""
    if not feature_requests:
        print("DEBUG: No feature requests provided to generate_feature_themes_with_llm")
        return {}
    
    print(f"DEBUG: Processing {len(feature_requests)} feature requests")
    
    # Prepare feature request texts for analysis
    request_texts = []
    for i, req in enumerate(feature_requests[:50]):  # Limit to top 50 for efficiency
        try:
            if isinstance(req, dict):
                if 'content' in req:
                    content = req['content']
                    sentiment_score = req.get('sentiment_score', 0)
                elif 'review' in req and isinstance(req['review'], dict):
                    content = req['review'].get('content', str(req))
                    sentiment_score = req['review'].get('sentiment_score', 0)
                else:
                    content = str(req)
                    sentiment_score = 0
            else:
                content = str(req)
                sentiment_score = 0
            
            sentiment_label = "positive" if sentiment_score > 0.3 else "neutral" if sentiment_score > -0.1 else "negative"
            request_texts.append(f"{i+1}. ({sentiment_label} user) {content}")
            
        except Exception as e:
            print(f"DEBUG: Error processing request {i}: {e}")
            request_texts.append(f"{i+1}. (unknown user) Error processing request")
    
    combined_requests = "\n".join(request_texts)
    print(f"DEBUG: Prepared {len(request_texts)} request texts for analysis")
    
    # Route to appropriate model
    if selected_model == 'Gemini-API':
        print("Routing to Gemini API for feature theme analysis...")
        prompt = f"""You are a Senior Product Manager analyzing user feature requests.
Analyze the following feature requests and identify the main themes/categories in 400 words.

For each theme:
1. Create a clear theme name (e.g., "Data Export & Backup")
2. List the request numbers that belong to this theme
3. Write a 2-3 sentence summary
4. Rate urgency as "High", "Medium", or "Low"

Feature requests:
{combined_requests}

Format:
**Theme Name**: App Interface [Request count: 3]
Summary: [2-3 sentence description]
Urgency: [High/Medium/Low]"""

        ai_response = generate_gemini_response(prompt)
        if ai_response:
            themes = parse_llm_feature_themes(ai_response, feature_requests)
            if themes:
                return themes
        
        # Fallback if Gemini fails
        print("DEBUG: Gemini API failed, using fallback")
        return generate_fallback_feature_themes(feature_requests)
    
    else:
        # Use local LLM
        print(f"Routing to local LLM ({selected_model}) for feature theme analysis...")
        models = get_models(selected_model)
        
        prompt = f"""<s>[INST] You are analyzing user feature requests.
Identify main themes from these requests in 400 words:

{combined_requests}

Format each theme as:
**Theme**: Name
Summary: Description
Urgency: High/Medium/Low
[/INST]
**Feature Analysis:**"""

        try:
            inputs = models['reasoning_tokenizer'](prompt, return_tensors="pt").to(models['device'])
            with torch.no_grad():
                outputs = models['reasoning_model'].generate(
                    **inputs, 
                    max_new_tokens=400, 
                    eos_token_id=models['reasoning_tokenizer'].eos_token_id, 
                    pad_token_id=models['reasoning_tokenizer'].eos_token_id
                )
            response_text = models['reasoning_tokenizer'].decode(outputs[0], skip_special_tokens=True)
            clear_gpu_cache()
            
            # Extract analysis
            if "**Feature Analysis:**" in response_text:
                analysis_text = response_text.split("**Feature Analysis:**")[-1].strip()
            else:
                analysis_text = response_text.split("[/INST]")[-1].strip()
            
            print(f"DEBUG: LLM Analysis Response: {analysis_text}")
            
            # Parse the response
            themes = parse_llm_feature_themes(analysis_text, feature_requests)
            if themes:
                return themes
            
        except Exception as e:
            print(f"DEBUG: Error in local LLM feature analysis: {e}")
        
        # Fallback for local LLM
        return generate_fallback_feature_themes(feature_requests)

def parse_llm_feature_themes(llm_response, feature_requests=None):
    """Parse LLM response to extract feature themes."""
    theme_sections = {}
    print("LLM Response for feature themes:")
    print(llm_response)
    
    if not feature_requests:
        print("ERROR: No feature_requests provided to parse_llm_feature_themes")
        return {}    
    
    # Extract theme sections from LLM response
    temp_sections = []
    
    # Pattern 1: **Theme Name**: Actual Theme Name format
    if '**' in llm_response:
        pattern = r'\*\*Theme Name\*\*:\s*([^\n]+)\n(.*?)(?=\*\*Theme Name\*\*:|$)'
        matches = re.findall(pattern, llm_response, re.DOTALL)
        
        for theme_name, theme_content in matches:
            theme_name = theme_name.strip()
            theme_content = theme_content.strip()
            if theme_name and theme_content:
                temp_sections.append((theme_name, theme_content))
        
        # Fallback to original pattern
        if not temp_sections:
            sections = re.split(r'\*\*([^*]+)\*\*', llm_response)
            for i in range(1, len(sections), 2):
                if i + 1 < len(sections):
                    theme_name = sections[i].strip()
                    theme_content = sections[i + 1].strip()
                    if theme_name and theme_content and theme_name != "Theme Name":
                        temp_sections.append((theme_name, theme_content))
    
    # Pattern 2: Number. Theme Name format
    if not temp_sections:
        sections = re.split(r'\d+\.\s*([^\n]+)', llm_response)
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                theme_name = sections[i].strip()
                theme_content = sections[i + 1].strip()
                if theme_name and theme_content:
                    temp_sections.append((theme_name, theme_content))
    
    # Pattern 3: Direct line-by-line parsing
    if not temp_sections:
        lines = llm_response.split('\n')
        current_theme = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this looks like a theme header
            if (line.endswith(':') or 
                any(keyword in line.lower() for keyword in ['theme', 'category', 'requests']) or
                re.match(r'^\d+\.', line)):
                
                # Save previous theme if exists
                if current_theme and current_content:
                    temp_sections.append((current_theme, '\n'.join(current_content)))
                
                current_theme = line.rstrip(':').strip('*').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Don't forget the last theme
        if current_theme and current_content:
            temp_sections.append((current_theme, '\n'.join(current_content)))

    print(f"DEBUG: Found {len(temp_sections)} theme sections")

    if not temp_sections:
        print("FALLBACK: Could not parse any themes, using fallback")
        return generate_fallback_feature_themes(feature_requests)
    
    # Process each theme section
    for theme_name, theme_content in temp_sections:
        print(f"DEBUG: Processing theme '{theme_name}'")
        
        # Extract summary
        summary_match = re.search(r'Summary:\s*(.*?)(?=\n\s*(Urgency|Request|$))', theme_content, re.DOTALL | re.IGNORECASE)
        if not summary_match:
            lines = [line.strip() for line in theme_content.split('\n') if line.strip() and not line.strip().startswith('Request') and not line.strip().startswith('Urgency')]
            summary_text = lines[0] if lines else theme_content[:250]
        else:
            summary_text = summary_match.group(1).strip().split('\n')[0].strip()
        
        # Extract urgency
        urgency_match = re.search(r'Urgency:\s*(.*?)(?=\n|$)', theme_content, re.IGNORECASE)
        urgency_text = urgency_match.group(1).strip() if urgency_match else "Medium"
        
        # Extract request numbers
        mapped_requests = []
        request_nums = []
        
        requests_match = re.search(r'Request\s+numbers?:?\s*\[?([\d,\s\-]+)\]?', theme_content, re.IGNORECASE)
        if requests_match:
            numbers_str = requests_match.group(1).strip().rstrip(',')
            request_nums = [int(num.strip()) for num in re.split(r'[,\s]+', numbers_str) if num.strip().isdigit()]
        
        if not request_nums:
            request_nums = [int(num) for num in re.findall(r'\b(\d+)\b', theme_content) 
                          if 1 <= int(num) <= len(feature_requests)]
        
        # Map numbers to actual requests
        for num in request_nums:
            if 1 <= num <= len(feature_requests):
                mapped_requests.append(feature_requests[num - 1])
        
        # If no specific numbers found, assign some requests as samples
        if not mapped_requests and len(feature_requests) > 0:
            num_samples = min(3, len(feature_requests))
            mapped_requests = feature_requests[:num_samples]
            print(f"DEBUG: No specific numbers found for '{theme_name}', using {num_samples} sample requests")
        
        request_count = len(mapped_requests)
        print(f"DEBUG: Theme '{theme_name}': Mapped {request_count} requests")

        # Store in theme_sections dict
        theme_sections[theme_name] = {
            'requests': mapped_requests,
            'summary': summary_text,
            'urgency': urgency_text,
            'count': request_count
        }

    print(f"DEBUG: Successfully parsed {len(theme_sections)} themes")
    return theme_sections
    
def generate_fallback_feature_themes(feature_requests):
    """Fallback method to generate feature themes when LLM fails."""
    if not feature_requests:
        return {}
    
    # Simple keyword-based grouping
    themes = {
        "User Interface & Design": {
            'requests': [],
            'summary': "Users requesting improvements to the app's visual design and user interface.",
            'urgency': 'Medium',
            'count': 0
        },
        "Features & Functionality": {
            'requests': [],
            'summary': "Users asking for new features or enhancements to existing functionality.",
            'urgency': 'High',
            'count': 0
        },
        "Performance & Reliability": {
            'requests': [],
            'summary': "Users requesting improvements to app performance and stability.",
            'urgency': 'Medium',
            'count': 0
        }
    }
    
    ui_keywords = ['design', 'interface', 'ui', 'theme', 'color', 'layout']
    perf_keywords = ['slow', 'fast', 'performance', 'speed', 'crash', 'bug']
    
    for req in feature_requests:
        content_lower = req['content'].lower()
        
        if any(keyword in content_lower for keyword in ui_keywords):
            themes["User Interface & Design"]['requests'].append(req)
        elif any(keyword in content_lower for keyword in perf_keywords):
            themes["Performance & Reliability"]['requests'].append(req)
        else:
            themes["Features & Functionality"]['requests'].append(req)
    
    # Update counts and remove empty themes
    for theme_name in list(themes.keys()):
        theme_data = themes[theme_name]
        theme_data['count'] = len(theme_data['requests'])
        if theme_data['count'] == 0:
            del themes[theme_name]
    
    print(f"Fallback themes generated: {[(name, data['count']) for name, data in themes.items()]}")
    return themes
        
def generate_category_summary(reviews, category_name, is_positive=False, selected_model=None):
    """Generate category summary using either Gemini API or local LLM."""
    # Universal fallback
    top_review_snippets = [f"• \"{truncate_review_content(r['content'], 180)}\"" for r in reviews[:2]]
    fallback_summary = f"Key themes mentioned by users include:\n" + "\n".join(top_review_snippets)

    if selected_model == "Gemini-API":
        print(f"Routing to Gemini API for {category_name} summary...")
        review_texts = "\n".join([f"- {r['content']}" for r in reviews[:15]])
        sentiment_type = "positive feedback" if is_positive else "user complaints"
        
        prompt = f"""Analyze the following user {sentiment_type} regarding "{category_name}".
Write a concise, one-paragraph summary (about 40-60 words) of the main themes.

User Reviews:
{review_texts}

Summary:"""
        
        ai_summary = generate_gemini_response(prompt)
        return ai_summary if ai_summary else fallback_summary

    else:
        print(f"Routing to Local LLM ({selected_model}) for {category_name} summary...")
        models = get_models(selected_model)
        if not reviews: 
            return "No reviews for this category."

        review_texts = "\n".join([f"- {r['content']}" for r in reviews[:10]])
        
        prompt = f"""Reviews about "{category_name}":
{review_texts}
Summary Keywords:"""
        
        response_text = generate_response(models['reasoning_model'], models['reasoning_tokenizer'], prompt, max_length=400)
        
        # Validate and parse the local model's response
        if "Summary Keywords:" in response_text:
            keywords_text = response_text.split("Summary Keywords:")[-1].strip()
            if keywords_text and len(keywords_text) > 5 and "reviews about" not in keywords_text.lower():
                keywords = re.split(r'[,\n-]', keywords_text)
                clean_keywords = [kw.strip('* "') for kw in keywords if len(kw.strip()) > 2]
                return f"Common themes identified: {', '.join(clean_keywords[:5])}."

        return fallback_summary
        
def summarize_with_llm(reviews, selected_model=None):
    """Generate AI insights using either Gemini API or local LLM."""
    # Universal fallback
    top_critical_reviews = sorted(reviews, key=lambda r: r.get('sentiment_score', 0))[:3]
    fallback_brief = "#### Top Critical Issues Identified\n\n"
    for i, review in enumerate(top_critical_reviews):
        fallback_brief += f"**{i+1}. Issue:** Users report problems such as: \"_{truncate_review_content(review['content'], 100)}_\"\n"
        fallback_brief += f"**Recommendation:** Investigate reports related to performance and feature stability.\n\n"
        
    if selected_model == "Gemini-API":
        print("Routing to Gemini API for AI Insights brief...")
        review_texts = "\n".join([f"- {r['content']}" for r in top_critical_reviews])
        prompt = f"""You are a Senior Product Analyst. Analyze these critical user reviews and create a product development brief.
Identify the top 2-3 most critical themes. For each theme, provide a one-sentence problem statement and two actionable recommendations.

User Reviews:
{review_texts}

Format your response in Markdown like this:
**1. Theme Name (e.g., Notification Failures)**
**Problem:** [One-sentence problem statement]
- **Recommendation:** [Actionable suggestion 1]
- **Recommendation:** [Actionable suggestion 2]"""

        ai_brief = generate_gemini_response(prompt)
        return markdown2.markdown(ai_brief) if ai_brief else markdown2.markdown(fallback_brief)

    else:
        print(f"Routing to Local LLM ({selected_model}) for AI Insights brief...")
        models = get_models(selected_model)
        if not reviews: 
            return markdown2.markdown("#### No critical reviews found.")

        review_texts = "\n".join([f"- {r['content']}" for r in top_critical_reviews])
        prompt = f"""Critical Reviews:
{review_texts}
Top 3 Problems:
1. """
        
        response_text = generate_response(models['reasoning_model'], models['reasoning_tokenizer'], prompt, max_length=80)
        
        # Validate and parse the local model's response
        if "Top 3 Problems:" in response_text:
            summary_text = response_text.split("Top 3 Problems:")[-1].strip()
            problems = [line.strip() for line in summary_text.split('\n') if len(line.strip()) > 5]
            if problems:
                ai_brief = "#### Top Critical Themes Identified by AI\n\n"
                for prob in problems:
                    ai_brief += f"- **Theme:** {re.sub(r'^\s*[\d\.\-\*]+\s*', '', prob)}\n"
                    ai_brief += f"- **Recommendation:** Prioritize investigation into this area.\n"
                return markdown2.markdown(ai_brief)

        return markdown2.markdown(fallback_brief)
        
def analyze_reviews_roberta(review_list, selected_model=None):
    """Analyze reviews using RoBERTa sentiment analysis with dynamic model selection."""
    models = get_models(selected_model)

    if not review_list:
        return {
            'avg_sentiment_score': 0, 'topics': {}, 'pain_points': {}, 'praise_points': {},
            'attention_reviews': [], 'praise_reviews': [], 'total_review_count': 0
        }

    # Filter out reviews with empty or invalid content
    valid_reviews = [r for r in review_list if r and isinstance(r.get('content'), str) and r.get('content').strip()]

    if not valid_reviews:
        return {
            'avg_sentiment_score': 0, 'topics': {}, 'pain_points': {}, 'praise_points': {},
            'attention_reviews': [], 'praise_reviews': [], 'total_review_count': 0
        }

    total_review_count = len(valid_reviews)
    texts = [review['content'] for review in valid_reviews]
    sentiments = []
    batch_size = 16

    print(f"Analyzing sentiment with model: {models.get('selected_model', 'Unknown')}...")

    # Process reviews in batches for sentiment analysis
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = models['classifier_tokenizer'](batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(models['device'])
        with torch.no_grad():
            outputs = models['classifier_model'](**inputs)
        scores = outputs.logits.softmax(dim=-1).cpu().numpy()
        sentiments.extend(scores)
        clear_gpu_cache()

    # Apply sentiment scores with enhanced detection
    for review, score in zip(valid_reviews, sentiments):
        base_score = score[2] - score[0]  # positive - negative
        content_lower = review['content'].lower()

        final_score = base_score
        is_hard_negative = False

        if any(keyword in content_lower for keyword in CRITICAL_ISSUES_SET):
            final_score = -0.95
            is_hard_negative = True
        elif any(indicator in content_lower for indicator in NEGATIVE_INDICATORS_SET):
            final_score = min(-0.6, base_score)
            is_hard_negative = True

        if not is_hard_negative and any(indicator in content_lower for indicator in POSITIVE_INDICATORS_SET):
            if base_score > 0:
                final_score = max(base_score, 0.7)
            else:
                final_score = base_score + 0.2

        review['sentiment_score'] = final_score
        review['display_content'] = review['content'][:250] + "..." if len(review['content']) > 250 else review['content']

    POSITIVE_THRESHOLD = 0.2
    NEGATIVE_THRESHOLD = -0.2

    # Initialize category mentions structure
    category_mentions = {}
    for main_cat, sub_topics in TOPIC_CATEGORIES.items():
        category_mentions[main_cat] = {
            'sub_topics': {sub_cat: {
                'total': 0, 'pos': 0, 'neg': 0, 
                'negative_reviews': [],
                'positive_reviews': []
            } for sub_cat in sub_topics},
            'main_total': 0, 'main_pos': 0, 'main_neg': 0, 'summary': '', 'positive_summary': ''
        }

    # Populate the positive/negative review lists for each category
    for review in valid_reviews:
        sentiment_score = review['sentiment_score']
        is_pos = sentiment_score > POSITIVE_THRESHOLD
        is_neg = sentiment_score < NEGATIVE_THRESHOLD
        
        for main_cat, sub_topics in TOPIC_CATEGORIES.items():
            for sub_cat, keywords in sub_topics.items():
                if any(keyword in review['content'].lower() for keyword in keywords):
                    stats = category_mentions[main_cat]['sub_topics'][sub_cat]
                    
                    stats['total'] += 1
                    if is_pos:
                        stats['pos'] += 1
                        stats['positive_reviews'].append(review)
                    elif is_neg:
                        stats['neg'] += 1
                        stats['negative_reviews'].append(review)
                    
                    break

    print("Generating topic summaries with correctly filtered reviews...")

    # Generate summaries using ONLY the relevantly filtered reviews
    for main_cat, data in category_mentions.items():
        data['main_pos'] = sum(sub['pos'] for sub in data['sub_topics'].values())
        data['main_neg'] = sum(sub['neg'] for sub in data['sub_topics'].values())
        data['main_total'] = data['main_pos'] + data['main_neg']
        
        # Collect ONLY NEGATIVE reviews for the Pain Point summary
        negative_summary_reviews = [r for sub in data['sub_topics'].values() for r in sub['negative_reviews']]
        negative_summary_reviews.sort(key=lambda r: r['sentiment_score'])
        
        # Collect ONLY POSITIVE reviews for the Praise Point summary
        positive_summary_reviews = [r for sub in data['sub_topics'].values() for r in sub['positive_reviews']]
        positive_summary_reviews.sort(key=lambda r: r['sentiment_score'], reverse=True)

        # Generate negative summary if there are enough negative reviews
        if data['main_neg'] >= 2:
            data['summary'] = generate_category_summary(negative_summary_reviews, main_cat, is_positive=False, selected_model=selected_model)
        
        # Generate positive summary if there are enough positive reviews
        if data['main_pos'] >= 2:
            data['positive_summary'] = generate_category_summary(positive_summary_reviews, main_cat, is_positive=True, selected_model=selected_model)

        # Populate display reviews from filtered lists
        for sub_cat_name, sub_cat_data in data['sub_topics'].items():
            sub_cat_data['negative_reviews_for_display'] = sorted(sub_cat_data['negative_reviews'], key=lambda r: r['sentiment_score'])[:3]
            sub_cat_data['positive_reviews_for_display'] = sorted(sub_cat_data['positive_reviews'], key=lambda r: r['sentiment_score'], reverse=True)[:3]
            
    # Define Pain Points and Praise Points based on majority sentiment
    pain_points = {
        cat_name: cat_data for cat_name, cat_data in category_mentions.items()
        if cat_data['main_neg'] > cat_data['main_pos'] and cat_data['main_neg'] > 0
    }
    
    praise_points = {
        cat_name: cat_data for cat_name, cat_data in category_mentions.items()
        if cat_data['main_pos'] > cat_data['main_neg'] and cat_data['main_pos'] > 0
    }

    # Calculate average sentiment score
    total_sentiment = sum(r['sentiment_score'] for r in valid_reviews)
    avg_sentiment_score = ((total_sentiment / len(valid_reviews)) + 1) * 2.5 if valid_reviews else 0

    # Get final lists of all positive/negative reviews
    all_attention_reviews = sorted([r for r in valid_reviews if r['sentiment_score'] < NEGATIVE_THRESHOLD],
                                 key=lambda r: r['sentiment_score'])
    all_praise_reviews = sorted([r for r in valid_reviews if r['sentiment_score'] > POSITIVE_THRESHOLD],
                               key=lambda r: r['sentiment_score'], reverse=True)

    print(f"Final stats: Total={total_review_count}, Negative={len(all_attention_reviews)}, Positive={len(all_praise_reviews)}")
    print(f"Identified {len(pain_points)} Pain Point categories and {len(praise_points)} Praise Point categories.")

    return {
        'avg_sentiment_score': round(avg_sentiment_score, 2),
        'topics': category_mentions,
        'pain_points': dict(sorted(pain_points.items(), key=lambda i: i[1]['main_neg'], reverse=True)),
        'praise_points': dict(sorted(praise_points.items(), key=lambda i: i[1]['main_pos'], reverse=True)),
        'attention_reviews': all_attention_reviews,
        'praise_reviews': all_praise_reviews,
        'total_review_count': total_review_count
    }
    
def generate_response(model, tokenizer, prompt, max_length=400):
    """Generic function to generate a response from a causal LM."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clear_gpu_cache()
    return response_text

def find_proof_reviews(reviews, category, limit=3):
    """Find proof reviews for a specific category."""
    proof = []
    keywords = []
    
    # Flatten keywords for the category
    for sublist in TOPIC_CATEGORIES.get(category, {}).values():
        keywords.extend(sublist)
    
    for review in reviews:
        content_lower = review['content'].lower()
        if any(kw in content_lower for kw in keywords):
            proof.append(review)
            if len(proof) >= limit:
                break
    
    return proof

def generate_structured_insights(analysis_present, analysis_previous, reviews_present_all, selected_model=None):
    """Generate structured insights with corrected model parameter passing."""
    # This call now correctly passes the selected_model parameter
    
    feature_summary = summarize_with_llm(analysis_present['attention_reviews'], selected_model)
    ai_summary = feature_summary
    print("AI Summary generated:")
    print(feature_summary)
    
    pp_present = analysis_present['pain_points']
    pp_previous = analysis_previous['pain_points']
    
    # Introduce a significance threshold
    significance_threshold = max(2, analysis_present.get('total_review_count', 100) // 50)
    
    # Identify different types of problems
    persisting_problems = {}
    for cat, data in pp_present.items():
        if cat in pp_previous:
            persisting_problems[cat] = {
                'count': data['main_neg'],
                'reviews': find_proof_reviews(analysis_present['attention_reviews'], cat)
            }
    
    newly_surfaced_problems = {}
    for cat, data in pp_present.items():
        if cat not in pp_previous:
            newly_surfaced_problems[cat] = {
                'count': data['main_neg'],
                'reviews': find_proof_reviews(analysis_present['attention_reviews'], cat)
            }
    
    resolved_problems = {}
    for cat, prev_data in pp_previous.items():
        if cat not in pp_present:
            current_neg = analysis_present['topics'].get(cat, {}).get('main_neg', 0)
            resolved_problems[cat] = {
                'prev': prev_data['main_neg'],
                'curr': current_neg,
                'reviews': find_proof_reviews(analysis_present.get('praise_reviews', []), cat)
            }

    # Enhanced feature request insights using LLM
    print("Analyzing feature requests...")
    
    feature_insights = {}
    feature_executive_summary = ""

    # Identify all potential feature requests from the entire review pool
    feature_requests = identify_feature_requests(reviews_present_all)
    
    print(f"Found {len(feature_requests)} potential feature requests.")
    feature_themes = {}
    if feature_requests:
        print("Generating feature themes with LLM...")
        if selected_model == "Gemini-API":
            print("Routing to Gemini API for feature theme generation...")
            feature_themes = generate_feature_themes_gemini(feature_requests)
        else:
            print(f"Routing to Local LLM ({selected_model}) for feature theme generation...")
            # Here you can call your old/simplified local LLM function
            feature_themes = generate_feature_themes_with_llm(feature_requests, selected_model) # Using the robust fallback for local models
    
    # Return structured insights
    return {
        "persisting_problems": persisting_problems,
        "newly_surfaced_problems": newly_surfaced_problems,
        "resolved_problems": resolved_problems,
        "feature_ideas": feature_themes,
        "ai_summary": feature_summary
    }

# ==============================================================================
# 5. DATA SCRAPING FUNCTIONS
# ==============================================================================

@force_close_connection
def scrape_reviews_until_date(app_id, target_date, max_reviews=50000):
    """Scrape reviews until a target date with optimized pagination."""
    all_reviews = []
    token = None
    
    while len(all_reviews) < max_reviews:
        try:
            result, token = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.NEWEST,
                count=200,
                continuation_token=token
            )
            
            if not result or not token:
                break
            
            all_reviews.extend(result)
            time.sleep(0.5)  # Rate limiting
            
            # Check if we've reached the target date
            if result[-1]['at'] < target_date:
                break
                
        except Exception as e:
            print(f"Error scraping reviews: {e}")
            break
    
    return all_reviews

@force_close_connection
def search_apps(brand_name, country='us'):
    """Search for apps with improved relevance scoring."""
    try:
        results = search(brand_name, n_hits=50, lang='en', country=country)
        term_lower = brand_name.lower()
        
        # Apply relevance scoring
        for r in results:
            title_lower = r.get('title', '').lower()
            if title_lower.startswith(term_lower):
                r['relevance_score'] = 3
            elif term_lower in title_lower:
                r['relevance_score'] = 2
            else:
                r['relevance_score'] = 1
        
        # Sort by relevance and rating
        sorted_results = sorted(
            results,
            key=lambda r: (r.get('relevance_score', 0), r.get('score') or 0),
            reverse=True
        )
        
        # Format results
        return [
            {
                'id': r['appId'],
                'text': f"{r['title']} (⭐ {r.get('score') or 0.0:.2f} | {format_installs(r.get('installs'))})"
            }
            for r in sorted_results
        ]
    
    except Exception as e:
        print(f"Error searching apps: {e}")
        return []

# ==============================================================================
# 6. FLASK ROUTES
# ==============================================================================

@app.route('/')
def index_page():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/search-apps', methods=['POST'])
def search_apps_route():
    """Handle app search requests."""
    try:
        data = request.json
        brand_name = data.get('brand_name')
        country = data.get('country', 'us')
        
        if not brand_name:
            return jsonify({'error': 'Brand name is required'}), 400
        
        results = search_apps(brand_name, country=country)
        return jsonify(results)
    
    except Exception as e:
        print(f"Error in /search-apps: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"An error occurred during search: {e}"}), 500

@app.route('/analyze', methods=['POST'])
def analyze_route():
    """Handle analysis requests with comprehensive error handling and model selection."""
    try:
        print("Starting analysis...")
        
        data = request.json
        app_id = data.get('app_id')
        date_range_1_str = data.get('date_range_1')
        date_range_2_str = data.get('date_range_2')
        selected_model = data.get('ai_model', 'microsoft/phi-2')
        
        # Corrected print statement for selected model
        if selected_model == 'Gemini-API':
            print(f"Selected AI Model: {selected_model} (cloud-based)")
        else:
            print(f"Selected AI Model: {selected_model} (local)")
        
        # Parse date ranges
        s1, e1 = parse_date_range_string(date_range_1_str)
        s2, e2 = parse_date_range_string(date_range_2_str)
        
        if not all([app_id, s1, e1, s2, e2]):
            return jsonify({'error': 'Invalid input parameters.'}), 400
        
        # Generate cache key for current request
        current_cache_key = generate_cache_key(app_id, date_range_1_str, date_range_2_str)
        global CACHE_KEY

        # Check if we need to clear cache (parameters changed)
        if CACHE_KEY is not None and CACHE_KEY != current_cache_key:
            clear_review_cache()

        # Check if reviews are already cached
        if current_cache_key in REVIEW_CACHE:
            print(f"Using cached reviews for {app_id} with date ranges {date_range_1_str} and {date_range_2_str}")
            cached_data = REVIEW_CACHE[current_cache_key]
            reviews_present_all = cached_data['present']
            reviews_previous_all = cached_data['previous']
        else:
            # Scrape new reviews
            oldest_target_date = min(s1, s2)
            print(f"Scraping reviews for {app_id} (not in cache)...")
            all_reviews = scrape_reviews_until_date(app_id, oldest_target_date)
    
            # Filter reviews by date ranges
            reviews_present_all = [
                r for r in all_reviews 
                if s1 <= r['at'] <= e1.replace(hour=23, minute=59)
            ]
            reviews_previous_all = [
                r for r in all_reviews 
                if s2 <= r['at'] <= e2.replace(hour=23, minute=59)
            ]
    
            # Cache the filtered reviews
            REVIEW_CACHE[current_cache_key] = {
                'present': reviews_present_all,
                'previous': reviews_previous_all
            }
            CACHE_KEY = current_cache_key
            print(f"Reviews cached for future use. Present: {len(reviews_present_all)}, Previous: {len(reviews_previous_all)}")



        print("Analyzing present reviews...")
        analysis_present = analyze_reviews_roberta(reviews_present_all, selected_model)
        
        print("Analyzing previous reviews...")
        analysis_previous = analyze_reviews_roberta(reviews_previous_all, selected_model)
        
        print("Generating insights...")
        insights = generate_structured_insights(
            analysis_present, 
            analysis_previous, 
            reviews_present_all,
            selected_model
        )
                
        print("Rendering results...")
        html_result = render_template(
            'results.html',
            analysis_present=analysis_present,
            analysis_previous=analysis_previous,
            insights=insights,
            count_present=len(reviews_present_all),
            count_previous=len(reviews_previous_all),
            form_data=data,
            selected_model=selected_model
        )
        
        return jsonify({'html': html_result})
    
    except Exception as e:
        print(f"Error in /analyze: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"A backend error occurred: {e}"}), 500

# ==============================================================================
# 7. APPLICATION ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    try:
        public_url = ngrok.connect(5000)
        print("=====================================================================================")
        print(f"✅ Your app is live! Access it here: {public_url}")
        print("=====================================================================================")
        app.run(port=5000)
    except Exception as e:
        print(f"\n❌ ERROR: An error occurred: {e}")
