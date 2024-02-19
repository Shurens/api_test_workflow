# Import des librairies
from unittest import TestCase, main
from fastapi.testclient import TestClient
import os
from api import app, model_1, model_2

# assertEqual(a, b) : Vérifie si a est égal à b.
# assertNotEqual(a, b) : Vérifie si a est différent de b.
        
# assertIn(a, b) : Vérifie si a est dans b.
# assertNotIn(a, b) : Vérifie si a n'est pas dans b.
        
# assertIs(a, b) : Vérifie si a est b.
# assertIsNot(a, b) : Vérifie si a n'est pas b.
        
# assertTrue(x) : Vérifie si x est vrai.
# assertFalse(x) : Vérifie si x est faux.
        
# assertIsNone(x) : Vérifie si x est None.
# assertIsNotNone(x) : Vérifie si x n'est pas None.
        
# assertIsInstance(a, b) : Vérifie si a est une instance de b.
# assertNotIsInstance(a, b) : Vérifie si a n'est pas une instance de b.
        
# assertRaises(exc, fun, *args, **kwargs) : Vérifie si fun(*args, **kwargs) lève une exception de type exc.
# assertRaisesRegex(exc, r, fun, *args, **kwargs) : Vérifie si fun(*args, **kwargs) lève une exception de type exc et dont le message correspond à l'expression régulière r.


# Tests unitaire de l'environnement de développement
class TestDev(TestCase):

    # Vérifie que les fichiers sont présents
    def test_files(self):
        import os
        list_files = os.listdir()
        self.assertTrue(len(list_files) > 0)

    # Vérifie que les requirements sont présents
    def test_requirements(self):
        self.assertTrue(os.path.isfile("requirements.txt"))

    
    # Vérifie que le gitignore est présent
    def test_gitignore(self):
        self.assertTrue(os.path.isfile(".gitignore"))
    

# Création du client de test
client = TestClient(app)

# Tests unitaire de l'API
class TestRoute(TestCase):

    # Vérifie que l'API est bien lancée

    # Vérifie le endpoint hello
    def test_hello(self):
        response = client.get("/hello")
        self.assertEqual(response.status_code, 200)
    
    # Vérifie le endpoint predict
    def test_predict(self):
        test_data = {"Gender": 1, "Age": 30, "Physical_Activity_Level": 2, "Heart_Rate": 80,
                     "Daily_Steps": 5000, "BloodPressure_high": 120, "BloodPressure_low": 80, "Sleep_Disorder": 0}
        response = client.post("/predict", json=test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())

    
# Test du modèle individuellement
class TestModel(TestCase):

    # Vérifie que le modèle est bien présent
    def test_model_presence(self):
        self.assertIsNotNone(model_1)
        self.assertIsNotNone(model_2)

    # Vérifie que le modèle est bien chargé
    def test_model_chargement(self):
        self.assertTrue(hasattr(model_1, 'predict'))
        self.assertTrue(hasattr(model_2, 'predict'))
    

# Démarrage des tests
if __name__== "__main__" :
    main(
        verbosity=2,
    )
