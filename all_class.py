import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def all_classification_models(X, y):
    from sklearn.naive_bayes import GaussianNB, BernoulliNB
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier 
    from sklearn.metrics import accuracy_score, confusion_matrix

    # Veri setini eğitim ve test olarak bölmek
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model listesi
    models = [
        GaussianNB(), BernoulliNB(), KNeighborsClassifier(), SVC(), 
        DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression(), XGBClassifier()
    ]
    model_names = [
        'GaussianNB', 'BernoulliNB', 'KNeighborsClassifier', 'SVC', 
        'DecisionTreeClassifier', 'RandomForestClassifier', 'LogisticRegression', 'XGBClassifier'
    ]
    
    # Doğruluk skorlarını saklamak için boş liste
    accuracies = []
    
    # Tüm modeller için döngü
    for model in models:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracies.append(accuracy_score(y_test, predictions))
    
    # Sonuçları DataFrame olarak saklamak
    results = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies
    }).sort_values(by='Accuracy', ascending=False)
    
    # En iyi modelin ismi
    best_model_name = results.iloc[0]['Model']
    
    # En iyi modeli bulma
    best_model_index = model_names.index(best_model_name)
    best_model = models[best_model_index]
    
    # En iyi modelin confusion matrisini çizme
    best_predictions = best_model.predict(x_test)
    cm = confusion_matrix(y_test, best_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title(f'Confusion Matrix of {best_model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
    return results

# Fonksiyonu kullanma örneği: all_classification_models(X, y)
