from transformers import pipeline

def main():
   # Path to the folder where we just saved the model
    model_path = "./best_mountain_ner_model"
    
    print("Uploading your trained model...")
   # We use a pipeline for Token Classification (NER)
    ner_pipeline = pipeline(
        "ner", 
        model=model_path, 
        tokenizer=model_path,
        aggregation_strategy="first" # Changed to "first" for nice subtoken gluing
    )
    print("The model is ready for testing!\n" + "="*40)
    
   # Examples to check
    test_sentences = [
        "Yesterday we decided to travel to Mount Everest, it was awesome.",
        "The highest peak in Ukraine is Hoverla, but Kilimanjaro is much higher.",
        "I want to climb Mont Blanc next summer with my friends."
    ]
    
    for sentence in test_sentences:
        print(f"\nТекст: {sentence}")
        predictions = ner_pipeline(sentence)
        
        if not predictions:
            print("Результат: Гір не знайдено.")
        else:
            print("Знайдені гори:")
            for entity in predictions:
                # Output the entity name, the entity itself, and the model confidence in %
                print(f"  - Гора: '{entity['word']}' (Впевненість: {entity['score']:.2%})")

if __name__ == "__main__":
    main()