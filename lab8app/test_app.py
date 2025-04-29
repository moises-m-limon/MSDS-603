import requests
import json
import sys

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Check Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_predict_endpoint():
    """Test the prediction endpoint with a sample patient"""
    print("\n=== Testing Single Prediction ===")
    
    # Sample data for a patient
    sample_patient = {
        "pregnancies": 6,
        "glucose": 148,
        "blood_pressure": 72,
        "skin_thickness": 35,
        "insulin": 0,
        "bmi": 33.6,
        "diabetes_pedigree_function": 0.627,
        "age": 50
    }
    
    print(f"Patient data: {json.dumps(sample_patient, indent=2)}")
    
    try:
        # Make the API call
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_patient,
            headers={"Content-Type": "application/json"}
        )
        
        # Print the results
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted Progression: {result['predicted_progression']:.4f}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Prediction request failed: {e}")
        return False

def test_batch_predict_endpoint():
    """Test the batch prediction endpoint with multiple patients"""
    print("\n=== Testing Batch Prediction ===")
    
    # Sample data for multiple patients
    batch_patients = [
        {
            "pregnancies": 6,
            "glucose": 148,
            "blood_pressure": 72,
            "skin_thickness": 35,
            "insulin": 0,
            "bmi": 33.6,
            "diabetes_pedigree_function": 0.627,
            "age": 50
        },
        {
            "pregnancies": 1,
            "glucose": 85,
            "blood_pressure": 66,
            "skin_thickness": 29,
            "insulin": 0,
            "bmi": 26.6,
            "diabetes_pedigree_function": 0.351,
            "age": 31
        },
        {
            "pregnancies": 8,
            "glucose": 183,
            "blood_pressure": 64,
            "skin_thickness": 0,
            "insulin": 0,
            "bmi": 23.3,
            "diabetes_pedigree_function": 0.672,
            "age": 32
        }
    ]
    
    print(f"Number of patients: {len(batch_patients)}")
    
    try:
        # Make the API call
        response = requests.post(
            f"{BASE_URL}/predict-batch",
            json=batch_patients,
            headers={"Content-Type": "application/json"}
        )
        
        # Print the results
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            results = response.json()["results"]
            print(f"Received {len(results)} predictions")
            
            # Print all predictions
            for i, result in enumerate(results):
                print(f"\nPatient {i+1}:")
                print(f"Predicted Progression: {result['predicted_progression']:.4f}")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Batch prediction request failed: {e}")
        return False

if __name__ == "__main__":
    # Check if API is healthy
    if not test_health():
        print("Health check failed, exiting")
        sys.exit(1)
    
    # Test with a sample patient
    test_predict_endpoint()
    
    # Test with multiple patients
    test_batch_predict_endpoint()