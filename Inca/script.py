from inca_framework import InCAFramework

if __name__ == "__main__":
    # Initialize and run the framework
    inca = InCAFramework()
    
    # Add classes
    inca.add_class("loan_application", ["Apply for loan", "Loan requirements"])
    inca.add_class("card_services", ["Report lost card", "Card replacement"])
    
    # Test prediction
    query = "How do I apply for a home loan?"
    print("Prediction result:", inca.predict(query))