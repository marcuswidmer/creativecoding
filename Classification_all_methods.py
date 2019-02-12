from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def trainClassifier(x_mat):
	# x_mat = matrix of size [row,col,features] with features from all pixels in ONE class
	mu_s = np.zeros(x_mat.shape[2])
	covar_s = np.zeros([x_mat.shape[2],x_mat.shape[2]])
	for row in range(x_mat.shape[0]):
		for col in range(x_mat.shape[1]):
			mu_s=mu_s+x_mat[row,col,:]
	mu_s=mu_s/(x_mat.shape[0]*x_mat.shape[1]) 
	for row in range(x_mat.shape[0]):
		for col in range(x_mat.shape[1]):
			covar_s=covar_s+(x_mat[row,col,:]-mu_s).reshape(x_mat.shape[2],1)@np.transpose((x_mat[row,col,:]-mu_s).reshape(x_mat.shape[2],1))
	covar_s=covar_s/(x_mat.shape[0]*x_mat.shape[1]) 

	return(mu_s,covar_s)
	
def meanDistanceClassfier(Components,classes,mu_s):
	Components = Components.reshape(num_components)
	num_classes = len(classes)
	dec_arr = np.zeros(num_classes)
	for Class in range(num_classes):
		#pm = freq_response.reshape(1,len_freq_resp)@cha_basis
		dec_arr[Class] = np.linalg.norm(Components-mu_s[:,Class])
	Class_choise=np.argmin(dec_arr)
	return Class_choise

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      #  print("Normalized confusion matrix")
    #else:
     #   print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def multivariateGuassianClassifier(freq_response,classes,mu_s,covar_s,prior_class_prob,num_features):
	len_freq_resp=len(freq_response)
	num_classes = len(classes)
	dec_arr = np.zeros(num_classes)
	for Class in range(num_classes):
		pm = freq_response.reshape(1,len_freq_resp)
		dec_arr[Class] = -np.log(2*np.pi)/2-np.log(np.linalg.det(covar_s[:,:,Class]))/2-np.transpose((pm-mu_s[:,Class]).reshape(num_features,1))@np.linalg.inv(covar_s[:,:,Class])@((pm-mu_s[:,Class]).reshape(num_features,1))/2+np.log(prior_class_prob[Class]) 
	Class_choise=np.argmax(dec_arr)
	norm_dec = dec_arr+np.abs(np.min(dec_arr))
	#print(norm_dec)
	#print('Classification result: %s' %(classes[Class_choise]))#,norm_dec[Class_choise]/np.sum(norm_dec)*100))
	return Class_choise

#-----------------------------------------------------------
# ---------------- PARAMS ----------------------------------
#-----------------------------------------------------------

dim_red = 'LDA' #'LDA','PCA'
classifier = 'knn' #'knn','mmd','mg'

classes = ['Stickleback','Whitefish','Copper_sphere']#,'Whitefish','Copper_sphere']
classes_2 = ['Stickleback','Whitefish','Copper sphere']
folder = 'LinearNormalisedData_160_260kHz' #'LinearNormalisedData_90_170kHz' #FFTfromChirp_Hanning_FourQuadrants_Data' 'ExtractedFeatures'

state = 'FreeSwimming'

scale_mean = True
scale_std = True
explained_variance = 0.9993 #0.9993
pca = PCA(n_components=explained_variance)
lda = LinearDiscriminantAnalysis(n_components=2)	

kk = 7

classes_3 = []
for i in range(len(classes)):
	classes_3 = np.append(classes_3,'%s (test)' %classes_2[i])

extracted_features = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
#extracted_features = np.array([14,16,13,4,1])
if folder == 'LinearNormalisedData_90_170kHz':
	freq_range = '90-170kHz'
elif folder == 'LinearNormalisedData_160_260kHz':
	freq_range = '160-260kHz'
else:
	freq_range = ''

feature = np.array(['Mean','Var','Skew','Kurtosis','TrendOverall','TrendStartMainPeak','TrendMainPeakStop','InclinationMinMax','InclinationMaxMin','TrendInterPeak','Number of Dynamic peaks','PeakIntensity Peak1','PeakIntensity Peak2','FFT1(Mean)','FFT2','FFT3','FFT4','FFT5'])

if folder == 'ExtractedFeatures':
	num_features = len(extracted_features)
	print(num_features)
	train_free = 100
	test_free = 50
elif folder == 'LinearNormalisedData_90_170kHz':
	num_features = 656
	train_free =400
	train_tethered = 200
	test_free = 60
	test_tethered = 40
elif folder == 'LinearNormalisedData_160_260kHz':
	num_features = 1092
	train_free =400
	train_tethered = 200
	test_free = 60
	test_tethered = 40



if state == 'Both':
	train_datas = train_free+train_tethered
	test_datas = test_free+test_tethered 
elif state == 'Tethered':
	train_datas = train_tethered
	test_datas = test_tethered
elif state == 'FreeSwimming':
	train_datas = train_free
	test_datas = test_free

state_orig = state

num_classes = len(classes)
prior_class_prob = np.ones(num_classes)/num_classes

train_response = np.zeros([num_classes,train_datas,num_features])
test_response = np.zeros([num_classes,test_datas,num_features])
temp2 = np.zeros([train_datas,num_features])
y=np.zeros(num_classes*train_datas)

#----------------READ TRAINING DATA-------------------
for Class in range(num_classes):
	if state_orig == 'Both':
		state = 'FreeSwimming'
		other_state_name = 'Tethered' 
		file_input='/users/marcuswidmer/Documents/UiO/Master/FrequencyClassification/%s/%s_%s_trainingset.txt' %(folder,classes[Class],other_state_name)
		temp2 = np.loadtxt(file_input, delimiter=',')
	file_input='/users/marcuswidmer/Documents/UiO/Master/FrequencyClassification/%s/%s_%s_trainingset.txt' %(folder,classes[Class],state)
	temp = np.loadtxt(file_input, delimiter=',')
	for i in range(train_datas):
		if state_orig == 'Both':
			train_response[Class,i,:]=(i<train_free)*temp[(i<train_free)*i,0:num_features]+(i>=train_free)*temp2[(i>=train_free)*(i-train_free),0:num_features]
		else:
			y[i+train_datas*Class]=Class
			if folder == 'ExtractedFeatures':
				for j in range(num_features):
					train_response[Class,i,j]=temp[i,extracted_features[j]]
			else:
				train_response[Class,i,:]=temp[i,:]

train_response = train_response.reshape(num_classes*train_datas,num_features)
scaler = StandardScaler(with_mean = scale_mean, with_std = scale_std)
train_response_z = scaler.fit_transform(train_response)  


if dim_red == 'PCA':
	Components = pca.fit_transform(train_response_z)	
	num_components = pca.n_components_
	mu_s = np.zeros([num_components,num_classes])
	covar_s = np.zeros([num_components,num_components,num_classes])
	x_mat = np.zeros([train_datas,num_components])
	print('The number of components is: %s' %num_components)	
	cha_basis = np.transpose(pca.components_[0:num_components,:])
elif dim_red == 'LDA':
	Components = lda.fit_transform(train_response_z,y)
	cha_basis = lda.scalings_
	num_components = num_classes-1
	mu_s = np.zeros([num_components,num_classes])
	covar_s = np.zeros([num_components,num_components,num_classes])
	x_mat = np.zeros([train_datas,num_components])

plt.plot(np.linspace(160,260,num_features),cha_basis[:,0])
plt.title('Weights for optimal separation, %s' %freq_range)
plt.xlabel('Frequency [kHz]')
plt.ylabel('Weight')
plt.show()

# -------------Setting up the kNN classifier ---------------------
neigh = KNeighborsClassifier(n_neighbors=kk)
neigh.fit(Components,y) 


# ------------ Plotting Contribution of each feature --------------------
if folder == 'ExtractedFeatures':
	most_contributing_features = cha_basis
	sorting_best_features = np.sort(np.sum(np.abs(most_contributing_features),axis=1))
	arg_sort = np.argsort(np.sum(np.abs(most_contributing_features),axis=1))	
	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel('Weight')
	if dim_red == 'PCA':
		ax.set_title('Contribution to variance', fontsize = 15)
	elif dim_red == 'LDA':
		ax.set_title('Contribution to separation', fontsize = 15)
	plt.barh(np.arange(num_features).astype(int),sorting_best_features,align='center')

	tick_marks = np.arange(num_features)
	plt.yticks(tick_marks, feature[extracted_features[arg_sort]], rotation=0)
	plt.grid()
	plt.show()

if folder == 'LinearNormalisedData_90_170kHz':
	fig = plt.figure(figsize = (16,8))
	ax = fig.add_subplot(1,2,1) 
	plt.title('Weights for optimal separation of classes')
	plt.plot(cha_basis[:,1])
	ax = fig.add_subplot(1,2,2)
	plt.bar(np.arange(num_features).astype(int),np.abs(np.fft.fft(np.hanning(num_features)*cha_basis[:,1])))
	plt.show()
for Class in range(num_classes):
	for i in range(train_datas):
		x_mat[i,:] = train_response_z[i+train_datas*Class,:].reshape(1,num_features)@cha_basis
	[mu_s[:,Class],covar_s[:,:,Class]]=trainClassifier(x_mat.reshape(1,train_datas,num_components))


# -----------READ TEST DATA ------------------------
for ActualClass in range(num_classes):
	if state_orig == 'Both':
		state = 'FreeSwimming'
		other_state_name = 'Tethered' 
		file_input='/users/marcuswidmer/Documents/UiO/Master/FrequencyClassification/%s/%s_%s_testset.txt' %(folder,classes[ActualClass],other_state_name)
		temp2 = np.loadtxt(file_input, delimiter=',')
	file_input='/users/marcuswidmer/Documents/UiO/Master/FrequencyClassification/%s/%s_%s_testset.txt' %(folder,classes[ActualClass],state)
	temp = np.loadtxt(file_input, delimiter=',')
	for i in range(test_datas):
		if state_orig == 'Both':
			test_response[ActualClass,i,:]=(i<test_free)*temp[(i<test_free)*i,0:num_features]+(i>=test_free)*temp2[(i>=test_free)*(i-test_free),0:num_features]
		else:
			y[i+test_datas*ActualClass]=ActualClass
			if folder == 'ExtractedFeatures':
				for j in range(num_features):
					test_response[ActualClass,i,j]=temp[i,extracted_features[j]]
			else:
				test_response[ActualClass,i,:]=temp[i,:]

tick = time.time()

test_response = test_response.reshape(num_classes*test_datas,num_features)
test_response_z = scaler.transform(test_response)
if dim_red == 'PCA':
	TestComponents = pca.transform(test_response_z)
elif dim_red == 'LDA':
	TestComponents = lda.transform(test_response_z)
	


y_test = np.zeros(num_classes*test_datas)
y_pred = np.zeros(num_classes*test_datas)

for ActualClass in range(num_classes):
	for i in range(test_datas):
		y_test[i+test_datas*ActualClass] = ActualClass
		if classifier == 'mmd':
			y_pred[i+test_datas*ActualClass] = meanDistanceClassfier(TestComponents[i+test_datas*ActualClass,:],classes,mu_s)
		elif classifier == 'mg':
			y_pred[i+test_datas*ActualClass] = multivariateGuassianClassifier(TestComponents[i+test_datas*ActualClass,:],classes,mu_s,covar_s,prior_class_prob,num_components)
if classifier == 'knn':
	y_pred = neigh.predict(TestComponents)

print(np.dot(train_response_z[2,:],cha_basis[:,1]))



fig = plt.figure(figsize = (12,6))
ax = fig.add_subplot(1,2,1) 
ax.set_xlabel('%s Component 1' %dim_red, fontsize = 15)
ax.set_ylabel('%s Component 2' %dim_red, fontsize = 15)
ax.set_title('%s Test data. %s' %(dim_red,freq_range), fontsize = 20)

colors = ['r', 'g', 'b']
colors2 = ['y','c','k']
"""for Class in range(num_classes):
	for i in range(train_datas):
		if i == 0:
			ax.scatter(Components[i+train_datas*Class,0]
					   , Components[i+train_datas*Class,1]
					   , c = colors[Class]
					   , label = classes_2[Class])
		else:
			ax.scatter(Components[i+train_datas*Class,0]
					   , Components[i+train_datas*Class,1]
					   , c = colors[Class])"""
for ActualClass in range(num_classes):
	for i in range(test_datas):
		if i == 0:
			ax.scatter(TestComponents[i+test_datas*ActualClass,0],TestComponents[i+test_datas*ActualClass,1]
				, c = colors2[ActualClass]
				, label = classes_3[ActualClass])
		else:
			ax.scatter(TestComponents[i+test_datas*ActualClass,0],TestComponents[i+test_datas*ActualClass,1]
			, c = colors2[ActualClass])

ax.legend()
ax.grid()

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
diag_cnf = np.diag(cnf_matrix)
print(dim_red)
print(classifier)
print('Train data: %s' %train_datas)
print('Test data: %s' %test_datas)
print(np.sum(diag_cnf)/(test_datas*num_classes))
print('Classification time: %s' %(time.time()-tick))
if dim_red == 'PCA':
	print('Explained variance (PCA): %s'%(explained_variance))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix

ax = fig.add_subplot(1,2,2) 
plot_confusion_matrix(cnf_matrix
					, normalize = True
					, classes=classes_2
					, title='Confusion matrix')

plt.show()
