
from transformers import pipeline

def generate_recipe(df):
    # Assuming each row in the dataframe `df` has a column 'ingredients' containing a list of ingredients
    # Initialize the text generation pipeline
    model_name = "gpt-3"  # Placeholder model name, use the appropriate model
    nlp = pipeline('text-generation', model=model_name)
    
    recipes = []  # A list to store the generated recipes
    titles = []  # A list to store the generated recipe titles
    cooking_times = []  # A list to store the generated cooking times
    
    for index, row in df.iterrows():
        # Format the ingredients list into a string
        ingredients_str = ', '.join(row['ingredients'])
        
        # Prepare the prompt for generating a recipe
        recipe_prompt = f"Given the ingredients {ingredients_str}, generate a cooking recipe."
        
        # Generate the recipe
        recipe_res = nlp(recipe_prompt, max_length=500, num_return_sequences=1)
        recipe = recipe_res[0]['generated_text']
        recipes.append(recipe)
        
        # Prepare the prompt for generating a recipe title
        title_prompt = f"Given the ingredients {ingredients_str}, generate a title for the cooking recipe."
        
        # Generate the title
        title_res = nlp(title_prompt, max_length=50, num_return_sequences=1)
        title = title_res[0]['generated_text']
        titles.append(title)
        
        # Prepare the prompt for estimating cooking time
        time_prompt = f"Given the recipe {recipe}, estimate the cooking time in minutes."
        
        # Generate the cooking time
        time_res = nlp(time_prompt, max_length=50, num_return_sequences=1)
        cooking_time = time_res[0]['generated_text']
        cooking_times.append(cooking_time)
    
    # Add the generated recipes, titles, and cooking times as new columns to the DataFrame
    df['recipe'] = recipes
    df['title'] = titles
    df['cooking_time'] = cooking_times
    
    return df
