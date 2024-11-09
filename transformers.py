from transformers import MarianMTModel, MarianTokenizer

def traducir(texto, modelo="Helsinki-NLP/opus-mt-en-es"):
    """
    Traduce un texto usando el modelo MarianMT.
    
    Parámetros:
    - texto: El texto a traducir.
    - modelo: El nombre del modelo a usar. Por defecto se utiliza un modelo de inglés a español.
    
    Retorna:
    - La traducción del texto en el idioma de destino.
    """
    # Cargar el tokenizador y el modelo de MarianMT
    tokenizer = MarianTokenizer.from_pretrained(modelo)
    model = MarianMTModel.from_pretrained(modelo)
    
    # Tokenización del texto original
    inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
    
    # Generación de la traducción
    outputs = model.generate(**inputs)
    
    # Decodificación de la salida del modelo
    traduccion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return traduccion

# Ejemplo de uso
texto_original = "Master in Systems Engineering at the National University of the Altiplano."
print(f"Texto original: {texto_original}")

# Traducir el texto original
traduccion = traducir(texto_original)
print(f"Traducción: {traduccion}")

# Traduciendo una lista de frases
textos = [
    "Hello, how are you?",
    "What is your name?",
    "I am learning to use transformers for translation.",
    "This is a test of machine translation."
]

# Traducir cada frase y mostrar el resultado
for texto in textos:
    print(f"\nTexto original: {texto}")
    traduccion = traducir(texto)
    print(f"Traducción: {traduccion}")

# Ver Vocabulario
# Cargar el tokenizador del modelo
modelo = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(modelo)

# Obtener el vocabulario
vocabulario = tokenizer.get_vocab()

# Mostrar algunas entradas del vocabulario
for token, id_token in list(vocabulario.items())[:100]:  # Muestra los primeros 10 tokens
    print(f"Token: {token} | ID: {id_token}")
