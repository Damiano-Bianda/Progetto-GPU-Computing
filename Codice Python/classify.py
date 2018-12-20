import csv
import sys
import math

def load_models(path):
	histograms = {}
	with open(path) as csvfile:
		rows = list(csv.reader(csvfile, delimiter = ','))
		header = rows[0]
		rows = rows[1:]
		
		for row in rows:
			label = row[0]
			histograms[label] = list(map(int, row[1:]))
	return histograms

def load_histograms(path):
	histograms = []
	with open(path) as csvfile:
		rows = list(csv.reader(csvfile, delimiter=','))
		header = rows[0]
		rows = rows[1:]

		for row in rows:
			label = row[0]
			histograms.append((label,list(map(int, row[1:]))))
	return histograms


def histogram_similarity(histogram1, histogram2):
	return sum( b1 * math.log(b2) for b1, b2 in zip(histogram1, histogram2))
	
	
def categorize(models, histogram_to_label):
	predicted_label = ""
	max_similarity_score = -1
	for label_model, histogram_model in models.items():
		current_similarity_score = histogram_similarity(histogram_to_label, histogram_model)
		if current_similarity_score > max_similarity_score:
			max_similarity_score = current_similarity_score
			predicted_label = label_model
	return predicted_label

def checkHistogramsFromFile(histograms, message):
	size = len(histograms[0])
	for  hist in histograms:
		if size != len(hist):
			print(message)
			exit(1)
	return size
	
if __name__=='__main__':

	if len(sys.argv) != 3:
		print('This program loads two csv files containing histograms and proceed to classification, first is reference, second histograms that need to be classified.')
		print('usage:\tprogramName "disk:\\absolute\\path\\reference_istograms_file" "disk:\\absolute\\path\\to_label_histograms_file"')
		sys.exit(0);
			
		
	models = load_models(sys.argv[1])
	
	modelHistogramSize = checkHistogramsFromFile(list(models.values()),'histograms in model in file have different sizes')
	
	print('Loaded {0} models: '.format(len(models)))
	for model in models:
		print('- {0}'.format(model))

	
	to_label = load_histograms(sys.argv[2])
	
	unlabelHistogramSize = checkHistogramsFromFile([elem[1] for elem in to_label],'unlabeled histograms in file have different sizes')
	
	if modelHistogramSize != unlabelHistogramSize:
		print('models and histograms have different size')
		exit(1)
	
	print('Loaded {0} histograms that must be labelled: '.format(len(to_label)))
	for true_label, histogram in to_label:
		print('- True label: {0}\t\tPredicted_label: {1}'.format(true_label ,categorize(models, histogram)))
