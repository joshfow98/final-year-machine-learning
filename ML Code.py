from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import pandas as pd
from time import process_time

def load_data_from_csv(input_csv):
    print('Parsing file ' + input_csv + '...')
    df = pd.read_csv(input_csv, header=0)
    csv_headings = list(df.columns.values)
    feature_names = csv_headings[:len(csv_headings) - 1]
    df = df._get_numeric_data()
    numpy_array = df.values
    number_of_rows, number_of_columns = numpy_array.shape
    instances = numpy_array[:, 0:number_of_columns - 1]
    labels = []
    for label in numpy_array[:, number_of_columns - 1:number_of_columns].tolist():
        labels.append(label[0])
    print('File ' + input_csv + ' parsed')
    return feature_names, instances, labels

mnb_classifier = MultinomialNB()
rf_classifier = RandomForestClassifier(n_estimators=1500)
svc_classifier = svm.SVC(gamma='scale')
mv_classifier = VotingClassifier(estimators=[('subc1', svc_classifier),
                                             ('subc2', rf_classifier),
                                             ('subc3', mnb_classifier)],
                                             voting='hard')

classifiers = {'MV': {'classifier': mv_classifier, 'precision': 0, 'recall': 0, 'f1_score': 0, 'training_time': 0, 'prediction_time': 0}}

csv_training_files = './csv/reviews_Video_Games_training.csv'
csv_test_files = './csv/reviews_Video_Games_test.csv'

training__feature_names, training_instances, training_labels = load_data_from_csv(csv_training_files)
test_feature_names, test_instances, test_labels = load_data_from_csv(csv_test_files)

for key, classifier_values in classifiers.items():
    classifier = classifier_values['classifier']
    f1_score = classifier_values['f1_score']

    print(key + ' training...')
    
    before_time = process_time()
    classifier.fit(training_instances, training_labels)
    after_time = process_time()
    classifier_values['training_time'] = after_time - before_time
    
    print(key + ' finished training')
    
    print(key + ' predicting values...')
    
    before_time = process_time()
    predicted_labels = classifier.predict(test_instances)
    after_time = process_time()
    classifier_values['prediction_time'] = after_time - before_time
    
    print(key + ' finished predicting')
    
    predicted = classification_report(test_labels, predicted_labels, output_dict=True)
    weighted_avg = predicted['weighted avg']
    
    precision = weighted_avg['precision']
    recall = weighted_avg['recall']
    f1_score = weighted_avg['f1-score']
        
    classifier_values['precision'] = precision
    classifier_values['recall'] = recall
    classifier_values['f1_score'] = f1_score
    
    
for key, classifier_values in classifiers.items():
    print(key + 
          ' Precision: ' + str(classifier_values['precision']) + 
          ' Recall: ' + str(classifier_values['recall']) + 
          ' F1-score : ' + str(classifier_values['f1_score']) +
          ' Training Time: ' + str(classifier_values['training_time']) +
          ' Prediction Time: ' + str(classifier_values['prediction_time']))