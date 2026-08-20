"""
Crop Advisor - Simple command-line tool
Loads the trained model and asks the user for their field conditions,
then gives a recommendation with confidence and explanation.
"""
import pandas as pd
import joblib

# Load the trained model and original dataset (needed for the "why" explanations)
model = joblib.load('crop_model.pkl')
df = pd.read_csv('Crop_recommendation.csv')

FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
FEATURE_LABELS = {
    'N': 'Nitrogen level in soil',
    'P': 'Phosphorus level in soil',
    'K': 'Potassium level in soil',
    'temperature': 'Temperature (Celsius)',
    'humidity': 'Humidity (%)',
    'ph': 'Soil pH',
    'rainfall': 'Rainfall (mm)'
}

def get_user_input():
    print("Enter your field's conditions:\n")
    values = {}
    for feature in FEATURES:
        while True:
            try:
                val = float(input(f"  {FEATURE_LABELS[feature]}: "))
                values[feature] = val
                break
            except ValueError:
                print("  Please enter a number.")
    return pd.DataFrame([values])

def explain_crop(field_input, crop):
    crop_avg = df[df['label'] == crop][FEATURES].mean()
    lines = []
    for col in FEATURES:
        your_val = field_input[col].values[0]
        typical = crop_avg[col]
        tolerance = 0.15 * typical if typical != 0 else 1
        if abs(your_val - typical) <= tolerance:
            lines.append(f"    OK   {col:12s}: matches typical range")
        elif your_val > typical:
            lines.append(f"    !!   {col:12s}: higher than typical ({typical:.1f})")
        else:
            lines.append(f"    !!   {col:12s}: lower than typical ({typical:.1f})")
    return lines

def recommend(field_input, confidence_threshold=0.5):
    probs = model.predict_proba(field_input)[0]
    crop_names = model.classes_
    top3_idx = probs.argsort()[-3:][::-1]
    top_crop = crop_names[top3_idx[0]]
    top_confidence = probs[top3_idx[0]]

    print("\n" + "=" * 55)
    if top_confidence >= confidence_threshold:
        print(f"RECOMMENDATION: {top_crop}  (confidence: {top_confidence*100:.0f}%)")
        print("This is a solid match for your field conditions.")
    else:
        print(f"NO CONFIDENT MATCH  (best guess only {top_confidence*100:.0f}% confident)")
        print("Your field conditions don't strongly match any single crop.")
        print("Closest options below, with mismatches shown, for you to judge:")
    print("=" * 55)

    for i in top3_idx:
        crop = crop_names[i]
        conf = probs[i]
        print(f"\n{crop} ({conf*100:.0f}% confidence)")
        for line in explain_crop(field_input, crop):
            print(line)
    print()

if __name__ == "__main__":
    print("=" * 55)
    print("  CROP ADVISOR - Prototype V1")
    print("=" * 55)
    print()
    field = get_user_input()
    recommend(field)
