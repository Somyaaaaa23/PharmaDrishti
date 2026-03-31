"""
Persona Generator for PharmaDrishti

Generates 10,000 diverse synthetic Indian healthcare personas
representing the full spectrum of India's demographic diversity.
"""

import json
import numpy as np
import random
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Indian states and their characteristics
STATES = {
    'Maharashtra': {'tier_1_prob': 0.4, 'tier_2_prob': 0.35, 'tier_3_prob': 0.25},
    'Karnataka': {'tier_1_prob': 0.35, 'tier_2_prob': 0.40, 'tier_3_prob': 0.25},
    'Tamil Nadu': {'tier_1_prob': 0.30, 'tier_2_prob': 0.45, 'tier_3_prob': 0.25},
    'Delhi': {'tier_1_prob': 0.90, 'tier_2_prob': 0.10, 'tier_3_prob': 0.0},
    'Gujarat': {'tier_1_prob': 0.25, 'tier_2_prob': 0.40, 'tier_3_prob': 0.35},
    'West Bengal': {'tier_1_prob': 0.30, 'tier_2_prob': 0.35, 'tier_3_prob': 0.35},
    'Uttar Pradesh': {'tier_1_prob': 0.15, 'tier_2_prob': 0.30, 'tier_3_prob': 0.55},
    'Rajasthan': {'tier_1_prob': 0.20, 'tier_2_prob': 0.35, 'tier_3_prob': 0.45},
    'Punjab': {'tier_1_prob': 0.25, 'tier_2_prob': 0.40, 'tier_3_prob': 0.35},
    'Telangana': {'tier_1_prob': 0.35, 'tier_2_prob': 0.40, 'tier_3_prob': 0.25},
}

# Cities by tier
TIER_1_CITIES = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
TIER_2_CITIES = ['Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Bhopal', 'Visakhapatnam', 'Patna', 'Vadodara', 'Ludhiana']
TIER_3_CITIES = ['Agra', 'Nashik', 'Faridabad', 'Meerut', 'Rajkot', 'Varanasi', 'Amritsar', 'Allahabad', 'Ranchi', 'Jodhpur']

# Diseases and their prevalence
DISEASES = {
    'Type 2 Diabetes': 0.12,
    'Hypertension': 0.25,
    'Asthma': 0.08,
    'Arthritis': 0.15,
    'Heart Disease': 0.06,
    'Thyroid Disorder': 0.10,
    'Chronic Kidney Disease': 0.04,
    'Obesity': 0.20,
    'Anemia': 0.18,
    'COPD': 0.05,
}

# Occupations by income level
OCCUPATIONS = {
    'Low': ['Daily Wage Worker', 'Farmer', 'Street Vendor', 'Domestic Worker', 'Security Guard'],
    'Lower-Middle': ['Shopkeeper', 'Auto Driver', 'Factory Worker', 'Clerk', 'Electrician', 'Plumber'],
    'Middle': ['Teacher', 'Nurse', 'Accountant', 'Sales Manager', 'Bank Employee', 'Government Officer'],
    'Upper-Middle': ['Software Engineer', 'Doctor', 'Lawyer', 'Business Owner', 'Senior Manager', 'Architect'],
    'High': ['CEO', 'Senior Doctor', 'Entrepreneur', 'Investment Banker', 'Corporate Executive', 'Consultant']
}

# Insurance types
INSURANCE_TYPES = [
    'Uninsured',
    'Government Scheme (Ayushman Bharat)',
    'Employer-Provided',
    'Private Insurance',
    'Government Employee Insurance',
    'Self-Purchased'
]

# Healthcare providers
HEALTHCARE_PROVIDERS = [
    'Government Hospital',
    'Private Hospital',
    'Private Clinic',
    'Pharmacy Only',
    'Community Health Center',
    'Corporate Hospital'
]

# Digital health adoption levels
DIGITAL_ADOPTION = ['None', 'Low', 'Medium', 'High']

# Indian first names
FIRST_NAMES = [
    'Rajesh', 'Priya', 'Amit', 'Sneha', 'Vikram', 'Anjali', 'Rahul', 'Pooja',
    'Suresh', 'Kavita', 'Arun', 'Deepika', 'Manoj', 'Neha', 'Sanjay', 'Ritu',
    'Ramesh', 'Sunita', 'Ashok', 'Meera', 'Vijay', 'Anita', 'Kiran', 'Swati',
    'Prakash', 'Shweta', 'Dinesh', 'Preeti', 'Ajay', 'Nisha', 'Ravi', 'Seema',
    'Mahesh', 'Rekha', 'Santosh', 'Geeta', 'Naveen', 'Asha', 'Pankaj', 'Suman',
    'Sachin', 'Madhuri', 'Nitin', 'Pallavi', 'Rohit', 'Divya', 'Vishal', 'Shruti',
    'Gaurav', 'Aarti', 'Manish', 'Jyoti', 'Anil', 'Vandana', 'Sunil', 'Kalpana'
]

# Last names
LAST_NAMES = [
    'Kumar', 'Singh', 'Sharma', 'Patel', 'Reddy', 'Nair', 'Iyer', 'Gupta',
    'Verma', 'Rao', 'Desai', 'Joshi', 'Mehta', 'Shah', 'Agarwal', 'Banerjee',
    'Chatterjee', 'Das', 'Ghosh', 'Mukherjee', 'Kapoor', 'Malhotra', 'Khanna', 'Bhatia'
]


def generate_city_tier(state):
    """Generate city tier based on state probabilities"""
    probs = STATES[state]
    tier = np.random.choice(
        ['Tier 1', 'Tier 2', 'Tier 3'],
        p=[probs['tier_1_prob'], probs['tier_2_prob'], probs['tier_3_prob']]
    )
    return tier


def generate_city(tier):
    """Generate city based on tier"""
    if tier == 'Tier 1':
        return random.choice(TIER_1_CITIES)
    elif tier == 'Tier 2':
        return random.choice(TIER_2_CITIES)
    else:
        return random.choice(TIER_3_CITIES)


def generate_income_level(tier, age):
    """Generate income level based on tier and age"""
    # Younger people tend to have lower income
    age_factor = min(1.0, (age - 18) / 30)
    
    if tier == 'Tier 1':
        # Tier 1 cities have higher income distribution
        income_probs = [0.10, 0.20, 0.35, 0.25, 0.10]
    elif tier == 'Tier 2':
        income_probs = [0.20, 0.30, 0.30, 0.15, 0.05]
    else:  # Tier 3
        income_probs = [0.35, 0.35, 0.20, 0.08, 0.02]
    
    # Adjust for age
    income_probs = [p * (0.5 + 0.5 * age_factor) if i >= 2 else p * (1.5 - 0.5 * age_factor) 
                    for i, p in enumerate(income_probs)]
    income_probs = [p / sum(income_probs) for p in income_probs]  # Normalize
    
    return np.random.choice(
        ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'],
        p=income_probs
    )
  

def generate_annual_income(income_level):
    """Generate annual income based on income level"""
    income_ranges = {
        'Low': (120000, 200000),
        'Lower-Middle': (200000, 500000),
        'Middle': (500000, 1000000),
        'Upper-Middle': (1000000, 2500000),
        'High': (2500000, 7600000)
    }
    min_income, max_income = income_ranges[income_level]
    return int(np.random.lognormal(np.log(min_income + max_income) / 2, 0.3))


def generate_chronic_diseases(age):
    """Generate chronic diseases based on age"""
    # Older people more likely to have chronic diseases
    age_factor = (age - 18) / 42  # 0 at 18, 1 at 60
    
    diseases = []
    for disease, base_prob in DISEASES.items():
        # Increase probability with age
        prob = base_prob * (0.5 + 1.5 * age_factor)
        if np.random.random() < prob:
            diseases.append(disease)
    
    return ', '.join(diseases) if diseases else 'None'


def generate_insurance(income_level, tier):
    """Generate insurance status based on income and tier"""
    if income_level in ['High', 'Upper-Middle']:
        return np.random.choice(
            ['Employer-Provided', 'Private Insurance', 'Self-Purchased'],
            p=[0.5, 0.3, 0.2]
        )
    elif income_level == 'Middle':
        return np.random.choice(
            ['Employer-Provided', 'Government Scheme (Ayushman Bharat)', 'Private Insurance', 'Uninsured'],
            p=[0.4, 0.3, 0.2, 0.1]
        )
    else:
        return np.random.choice(
            ['Government Scheme (Ayushman Bharat)', 'Uninsured', 'Employer-Provided'],
            p=[0.5, 0.4, 0.1]
        )


def generate_insurance_sum(insurance_status, income):
    """Generate insurance sum assured based on insurance type and income"""
    if insurance_status == 'Uninsured':
        return 0
    elif 'Government Scheme' in insurance_status:
        return 500000  # Ayushman Bharat coverage
    elif insurance_status == 'Employer-Provided':
        return int(income * np.random.uniform(2, 5))
    else:  # Private or Self-Purchased
        return int(income * np.random.uniform(3, 10))


def generate_persona(index):
    """Generate a single persona"""
    # Basic demographics
    age = int(np.random.beta(2, 2) * 42 + 18)  # Age 18-60, bell curve
    gender = random.choice(['Male', 'Female'])
    state = random.choice(list(STATES.keys()))
    tier = generate_city_tier(state)
    city = generate_city(tier)
    
    # Economic
    income_level = generate_income_level(tier, age)
    annual_income = generate_annual_income(income_level)
    occupation = random.choice(OCCUPATIONS[income_level])
    
    # Healthcare
    chronic_diseases = generate_chronic_diseases(age)
    no_of_conditions = len(chronic_diseases.split(', ')) if chronic_diseases != 'None' else 0
    
    # Calculate monthly healthcare spend (2-5% of monthly income, higher if chronic diseases)
    monthly_income = annual_income / 12
    base_spend_pct = 0.02 + (0.03 * no_of_conditions / 5)  # 2-5% based on conditions
    monthly_healthcare_spend = int(monthly_income * base_spend_pct)
    
    # Insurance
    insurance_status = generate_insurance(income_level, tier)
    insurance_sum = generate_insurance_sum(insurance_status, annual_income)
    
    # Last doctor visit (more recent if chronic diseases)
    if no_of_conditions > 0:
        last_visit = int(np.random.exponential(2))  # Average 2 months
    else:
        last_visit = int(np.random.exponential(6))  # Average 6 months
    
    # Price sensitivity (inversely related to income)
    income_percentile = ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'].index(income_level) / 4
    price_sensitivity = 1.0 - income_percentile + np.random.normal(0, 0.1)
    price_sensitivity = max(0.0, min(1.0, price_sensitivity))
    
    # Healthcare provider preference
    if income_level in ['High', 'Upper-Middle']:
        provider = np.random.choice(
            ['Private Hospital', 'Corporate Hospital', 'Private Clinic'],
            p=[0.5, 0.3, 0.2]
        )
    elif income_level == 'Middle':
        provider = np.random.choice(
            ['Private Hospital', 'Private Clinic', 'Government Hospital'],
            p=[0.4, 0.4, 0.2]
        )
    else:
        provider = np.random.choice(
            ['Government Hospital', 'Community Health Center', 'Pharmacy Only'],
            p=[0.5, 0.3, 0.2]
        )
    
    # Digital health adoption (higher in younger, urban, educated)
    digital_score = (1 - age / 60) * 0.4  # Age factor
    digital_score += (['Tier 1', 'Tier 2', 'Tier 3'].index(tier) / -2 + 1) * 0.3  # Tier factor
    digital_score += income_percentile * 0.3  # Income factor
    digital_score += np.random.normal(0, 0.1)
    digital_score = max(0.0, min(1.0, digital_score))
    
    if digital_score > 0.7:
        digital_adoption = 'High'
    elif digital_score > 0.4:
        digital_adoption = 'Medium'
    elif digital_score > 0.2:
        digital_adoption = 'Low'
    else:
        digital_adoption = 'None'
    
    # Generate name
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    name = f"{first_name} {last_name} {index}"
    
    return {
        'name': name,
        'age': age,
        'gender': gender,
        'city': city,
        'state': state,
        'city_tier': tier,
        'occupation': occupation,
        'annual_income_inr': annual_income,
        'income_level': income_level,
        'monthly_healthcare_spend_inr': monthly_healthcare_spend,
        'insurance_status': insurance_status,
        'insurance_sum_assured_inr': insurance_sum,
        'chronic_diseases': chronic_diseases,
        'no_of_chronic_conditions': no_of_conditions,
        'last_doctor_visit_months_ago': last_visit,
        'price_sensitivity_score': round(price_sensitivity, 2),
        'preferred_healthcare_provider': provider,
        'digital_health_adoption': digital_adoption
    }


def generate_personas(n=10000):
    """Generate n personas"""
    print(f"Generating {n:,} diverse Indian healthcare personas...")
    print("This may take a few minutes...")
    
    personas = []
    for i in range(n):
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1:,} personas...")
        personas.append(generate_persona(i + 1))
    
    print(f"✓ Generated {n:,} personas successfully!")
    return personas


def save_personas(personas, filepath='data/indian_healthcare_personas.json'):
    """Save personas to JSON file"""
    # Create data directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(personas, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(personas):,} personas to {filepath}")
    
    # Print statistics
    print("\nPersona Statistics:")
    print(f"  Age range: {min(p['age'] for p in personas)}-{max(p['age'] for p in personas)} years")
    print(f"  Income range: ₹{min(p['annual_income_inr'] for p in personas):,} - ₹{max(p['annual_income_inr'] for p in personas):,}")
    
    # Tier distribution
    tier_counts = {}
    for p in personas:
        tier_counts[p['city_tier']] = tier_counts.get(p['city_tier'], 0) + 1
    print(f"\n  City Tier Distribution:")
    for tier, count in sorted(tier_counts.items()):
        print(f"    {tier}: {count:,} ({count/len(personas)*100:.1f}%)")
    
    # Income level distribution
    income_counts = {}
    for p in personas:
        income_counts[p['income_level']] = income_counts.get(p['income_level'], 0) + 1
    print(f"\n  Income Level Distribution:")
    for level in ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']:
        count = income_counts.get(level, 0)
        print(f"    {level}: {count:,} ({count/len(personas)*100:.1f}%)")
    
    # Chronic disease prevalence
    disease_count = sum(1 for p in personas if p['chronic_diseases'] != 'None')
    print(f"\n  Personas with chronic diseases: {disease_count:,} ({disease_count/len(personas)*100:.1f}%)")
    
    # Insurance coverage
    insured_count = sum(1 for p in personas if p['insurance_status'] != 'Uninsured')
    print(f"  Insured personas: {insured_count:,} ({insured_count/len(personas)*100:.1f}%)")


def main():
    """Main function"""
    print("=" * 60)
    print("PharmaDrishti - Persona Generator")
    print("=" * 60)
    print()
    
    # Generate personas
    personas = generate_personas(n=10000)
    
    # Save to file
    save_personas(personas)
    
    print("\n" + "=" * 60)
    print("Persona generation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review generated personas in data/indian_healthcare_personas.json")
    print("2. Run: python train_model.py")
    print("3. This will create 50 million interactions (10,000 × 5,000)")
    print("4. Training will take 15-30 minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
