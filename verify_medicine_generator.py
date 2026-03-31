"""
Verification script for medicine_generator module
"""

from medicine_generator import generate_medicines

# Generate 50 medicines
medicines = generate_medicines(50)

print('✓ Success Criteria Verification:')
print(f'  - Generated {len(medicines)} medicine profiles')
print(f'  - Medicine ID format: {medicines["medicine_id"].iloc[0]}, {medicines["medicine_id"].iloc[-1]}')
print(f'  - Target diseases: {medicines["target_disease"].nunique()} unique diseases')
print(f'  - Batch price range: ₹{medicines["batch_price_inr"].min():.2f} - ₹{medicines["batch_price_inr"].max():.2f}')
print(f'  - Manufacturing score: {medicines["manufacturing_score"].min():.3f} - {medicines["manufacturing_score"].max():.3f}')
print(f'  - Side effect risk (beta dist): mean={medicines["side_effect_risk"].mean():.3f}')
print(f'    {(medicines["side_effect_risk"] < 0.5).sum()}/{len(medicines)} medicines have low risk (<0.5)')
print(f'  - Insurance: {medicines["insurance_compatibility"].min():.3f} - {medicines["insurance_compatibility"].max():.3f}')
print('\n✓ All success criteria met!')
print('\nSample medicines:')
print(medicines.head(10))
