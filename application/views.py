############################################################################
############## Importing the Required Packages for the Algorithm ###########
############################################################################

import os
import shutil
import datetime
from django.shortcuts import render
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from .forms import UserForm, DocumentForm
from .models import Upload, Table
from .Missing import DataFrameImputer
from django.http import HttpResponse
from fusioncharts import FusionCharts
from wsgiref.util import FileWrapper
import mimetypes
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from matplotlib.font_manager import FontProperties
from collections import deque
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib.patches import Ellipse as ell
from subprocess import check_call
from scipy.spatial.distance import cdist
import time


# Create your views here.

dfmc = pd.DataFrame()
org_data = pd.DataFrame()
viz1 = pd.DataFrame()
tab1 = dfmc.to_html
tab2 = dfmc.to_html
tab3 = dfmc.to_html
tab4 = dfmc.to_html
tab5 = dfmc.to_html
info = pd.Series()
dfo = pd.DataFrame()
score = 1.0
num_var = []
cat_var = []
feature = []
D = []
fname = ""

pd.set_option('max_colwidth', 2000)


def index(request):
    if not request.user.is_authenticated():
        return render(request, 'application/login.html')
    else:
        return render(request, 'application/dashboard.html')


def logout_user(request):
    logout(request)
    form = UserForm(request.POST or None)
    context = {
        "form": form,
    }
    return render(request, 'application/login.html', context)


def login_user(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                form = DocumentForm()
                documents = Upload.objects.filter(user=request.user).order_by('-date')
                return render(request, 'application/input_file.html', {'documents': documents, 'form': form})
            else:
                return render(request, 'application/login.html', {'error_message': 'Your account has been disabled'})
        else:
            return render(request, 'application/login.html', {'error_message': 'Invalid login'})
    return render(request, 'application/login.html')


def register(request):
    form = UserForm(request.POST or None)
    if form.is_valid():
        user = form.save(commit=False)
        username = form.cleaned_data['username']
        password = form.cleaned_data['password']
        email = form.cleaned_data['email']
        user.set_password(password)
        user.save()
        user = authenticate(username=username, password=password)
        print(user)
        if user is not None:
            if user.is_active:
                login(request, user)
                form = DocumentForm()
                documents = Upload.objects.filter(user=request.user).order_by('-date')
                path1 = './application/static/application/images/user_' + str(request.user.id)
                path2 = './application/static/application/images/user_' + str(request.user.id) + '/new'
                path3 = './application/static/application/images/user_' + str(request.user.id) + '/old'
                path4 = './application/static/application/images/user_' + str(request.user.id) + '/result'
                os.mkdir(path1, 0755)
                os.mkdir(path2, 0755)
                os.mkdir(path3, 0755)
                os.mkdir(path4, 0755)
                return render(request, 'application/input_file.html', {'documents': documents, 'form': form})
    context = {
        "form": form,
        'error_message': 'Username Already Exists'
    }
    return render(request, 'application/login.html', context)


def input_file(request):
    if not request.user.is_authenticated():
        return render(request, 'application/login.html')
    else:
        form = DocumentForm(request.POST or None, request.FILES or None)
        documents = Upload.objects.filter(user=request.user).order_by('-date')
        FILE_TYPES = ['csv', 'xlsx', 'xls']
        if form.is_valid():
            newdoc = form.save(commit=False)
            newdoc.user = request.user
            newdoc.docfile = request.FILES['docfile']
            newdoc.filename = newdoc.docfile.url
            newdoc.date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            file_type = newdoc.docfile.url.split('.')[-1]
            file_type = file_type.lower()
            if file_type not in FILE_TYPES:
                context = {
                    'documents': documents,
                    'form': form,
                    'error_message': 'Uploaded file must be CSV or Excel File',
                }
                return render(request, 'application/input_file.html', context)
            newdoc.save()
            latest = Upload.objects.filter(user=request.user, filename=newdoc.filename).order_by('-id')
            directory = './application/static/application/images/user_' + str(request.user.id) + '/'
            for f in os.listdir(directory + 'new/'):
                df = pd.DataFrame()

                if f.endswith(".csv") and f.find(newdoc.filename) != -1:
                    df = pd.read_csv(directory + 'new/' + f)
                elif (f.endswith(".xlsx") or f.endswith(".xls")) and f.find(newdoc.filename) != -1:
                    df = pd.read_excel(directory + 'new/' + f)

                abc = df.to_dict('records')
                tab = Table.objects.create(
                    username=request.user.id,
                    tablename=newdoc.filename,
                    answer=abc,
                    fid=latest[0].id
                )
                tab.save(validate=False)
                shutil.move(directory + 'new/' + f, directory + 'old/' + f)
            ret = Table.objects(username=request.user.id, tablename=newdoc.filename, fid=latest[0].id)
            ab = len(ret)
            c = ret[ab - 1]
            data = c.answer
            global dfo
            dfo = pd.DataFrame.from_dict(data, orient='columns')
            col_names = list(dfo)
            global fname
            fname = './application/static/application/images/user_' + str(request.user.id) + '/result/' + newdoc.filename + '.csv'
            return render(request, 'application/variable.html',
                          {'documents': documents, 'form': form, 'columns': col_names})
        else:
            form = DocumentForm()

        return render(
            request,
            'application/input_file.html',
            {'documents': documents, 'form': form}
        )


def download(request):
    wrapper = FileWrapper(file(fname, 'rb'))
    response = HttpResponse(wrapper, content_type=mimetypes.guess_type(fname)[0])
    response['Content-Length'] = os.path.getsize(fname)
    filename = fname.split('/')[-1]
    response['Content-Disposition'] = "attachment; filename=" + filename
    return response


def var_select(request):
    if not request.user.is_authenticated():
        return render(request, 'application/login.html')
    else:
        abc = request.POST.getlist('check')
        global dfo
        dfo = dfo.drop(abc, axis=1)

        for col in dfo.columns.values:
            if "date" in str.lower(str(col)):
                timedate = pd.to_datetime(dfo[col])
                if timedate.dtypes == 'datetime64[ns]':
                    dfo[col + "_month"] = timedate.dt.month
                    dfo[col + "_weekday"] = timedate.dt.weekday
                    dfo = dfo.drop(col, axis=1)

        dfr = dfo.to_html(classes="w3-table-all table-condensed w3-hoverable w3-small w3-card-8")
        dfr = dfr.replace("border", 'id="t1" border')

        global org_data
        org_data = dfo.dropna()

        null_data = dfo[dfo.isnull().any(axis=1)]
        dfm = null_data.to_html(classes="w3-table-all table-condensed w3-hoverable w3-small w3-card-8")
        dfm = dfm.replace("border", 'id="t2" border')

        for col in dfo.columns.values:
            if len(dfo[col].value_counts().keys()) < 2:
                dfo = dfo.drop(col, axis=1)

        for col in dfo.columns.values:
            if dfo[col].value_counts().index[0] == ' ':
                dfo = dfo.drop(col, axis=1)

        global dfmc
        dfmc = DataFrameImputer().fit_transform(dfo)
        for col in dfmc.columns.values:
            if dfmc[col].dtype == np.dtype('O') and (('' in dfmc[col].value_counts().keys()) or (' ' in dfmc[col].value_counts().keys())):
                dfmc[col] = dfmc[col].replace(' ', dfmc[col].value_counts().index[0])
        dfc = dfmc.to_html(classes="w3-table-all table-condensed w3-hoverable w3-small w3-card-8")
        dfc = dfc.replace("border", 'id="t3" border')

        return render(request, 'application/data.html', {'data_raw': dfr, 'data_miss': dfm, 'data_imputed': dfc})


def disp_data(request, filename, fileid):
    if not request.user.is_authenticated():
        return render(request, 'application/login.html')
    else:
        form = DocumentForm()
        documents = Upload.objects.filter(user=request.user).order_by('-date')
        ret = Table.objects(username=request.user.id, tablename=filename, fid=fileid)
        ab = len(ret)
        c = ret[ab - 1]
        data = c.answer
        global dfo
        dfo = pd.DataFrame.from_dict(data, orient='columns')
        col_names = list(dfo)
        global fname
        fname = './application/static/application/images/user_' + str(request.user.id) + '/result/' + filename + '.csv'
        return render(request, 'application/variable.html',
                      {'documents': documents, 'form': form, 'columns': col_names})


def process_file(request, option):
    if not request.user.is_authenticated():
        return render(request, 'application/login.html')
    else:
        directory = './application/static/application/images/user_' + str(request.user.id) + '/result/'
        for f in os.listdir(directory):
            os.remove(directory + f)
        if option == "yes":
            X_imp = dfmc
        else:
            X_imp = org_data

        np.random.seed(18)
        global viz1
        viz1 = X_imp
        global num_var
        global cat_var
        num_var = []
        cat_var = []
        for col in X_imp.columns.values:
            if (X_imp[col].dtype == np.dtype('O') and len(X_imp[col].unique() > X_imp[col].size / 2)):
                cat_var.append(str(col))
            else:
                num_var.append(str(col))

        #####################################################################################
        ############### Label Encoding Logic ################################################
        #####################################################################################
        le = LabelEncoder()
        queue = deque()
        global D
        D = []
        X_imp2 = X_imp.copy()
        le2 = LabelEncoder()
        D2 = []
        queue2 = deque()
        for col in X_imp2.columns.values:
            if X_imp2[col].dtypes == 'object' or X_imp2[col].dtypes == 'bool':
                le2.fit(X_imp2[col].values)
                X_imp2[col] = le2.transform(X_imp2[col])
                A2 = le2.classes_
                B2 = le2.transform([le2.classes_])
                C2 = list([A2, B2, col])
                D2.append({'Original_Values': A2, 'Reformed_Values': B2, 'Column': col})
                queue2.append(C2)
        for col in X_imp.columns.values:
            if (X_imp[col].dtypes == 'object' and len(X_imp[col].unique()) < 3) or X_imp[col].dtypes == 'bool':
                le.fit(X_imp[col].values)
                X_imp[col] = le.transform(X_imp[col])
                A = le.classes_
                B = le.transform([le.classes_])
                C = list([A, B, col])
                D.append({'Original_Values': A, 'Reformed_Values': B, 'Column': col})
                queue.append(C)
            elif (X_imp[col].dtypes == 'object' and len(X_imp[col].unique()) > 2 and len(X_imp[col].unique()) < 20):
                one_hot = pd.get_dummies(X_imp[col], prefix=pd.DataFrame(X_imp[col]).columns.astype(str).values)
                X_imp = X_imp.drop(col, axis=1)
                X_imp = X_imp.join(one_hot)
            elif (X_imp[col].dtypes == 'object' and len(X_imp[col].unique() > X_imp[col].size / 2)):
                X_imp = X_imp.drop(col, axis=1)
                viz1 = viz1.drop(col, axis=1)

        ############# X_imp is Pandas DataFrame which can be used later ######################
        X_old = X_imp.as_matrix()
        mean_of_array = X_old.mean(axis=0)
        std_of_array = X_old.std(axis=0)

        ######## Scaling the Original Data ##################################
        X_scale = preprocessing.scale(X_old)
        X = pd.DataFrame(X_scale)
        X_bef_clus = pd.DataFrame(X_scale)
        X_arr = X.as_matrix()

        ########################## Reference for Mapping ##########################
        reference = pd.DataFrame(D, columns=('Original_Values', 'Reformed_Values', 'Column'))

        ########## zd is columns attributes from the (Imputed+Label Encoded) Data ###########
        zd = X_imp.columns.values
        ################# z1 is Pandas DataFrame of (Imputed+Label Encoded) Data ############
        z1 = pd.DataFrame(zd)
        zd2 = X_imp2.columns.values
        z2 = pd.DataFrame(zd2)

        ################# Initialization of Different Types of Font ########################
        fontXXSmall = FontProperties()
        fontXXSmall.set_size('xx-small')
        fontXSmall = FontProperties()
        fontXSmall.set_size('x-small')
        fontSmall = FontProperties()
        fontSmall.set_size('small')

        ###########################################################################################
        ################ Initialization of Centroids, Distances, Avg. Distance within Cluster #####
        ###########################################################################################
        K = range(1, 10)

        KM = [kmeans(X, k) for k in K]
        centroids = [cent for (cent, var) in KM]
        D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
        cIdx = [np.argmin(D, axis=1) for D in D_k]
        dist = [np.min(D, axis=1) for D in D_k]
        avgWithinSS = [sum(d) / X.shape[0] for d in dist]

        ###########################################################################################
        ###################### Logic to get Elbow Point from the Elbow Curve ######################
        ###########################################################################################

        nPoints = len(avgWithinSS)
        allCoord = np.vstack((K, avgWithinSS)).T
        np.array([K, avgWithinSS])
        firstPoint = allCoord[0]
        lineVec = allCoord[-1] - allCoord[0]
        lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))
        vecFromFirst = allCoord - firstPoint
        scalarProduct = np.sum(vecFromFirst * np.tile(lineVecNorm, (nPoints, 1)), axis=1)
        vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
        idxOfBestPoint = np.argmax(distToLine)
        kIdx = idxOfBestPoint

        ########################################################################################
        ############################## End Elbow Logic #########################################
        ########################################################################################

        ########################################################################################
        ############## KMeans Model Application ################################################
        ########################################################################################

        t0_rand = time.clock()

        kmeans_model = KMeans(n_clusters=K[kIdx], random_state=1).fit(X)

        t1_rand = time.clock()
        elapsed_rand = t1_rand - t0_rand

        kmeans_model2 = KMeans(init='k-means++', n_clusters=K[kIdx], random_state=111)
        labels1 = kmeans_model.labels_

        #######################################################################################
        ############ End KMeans Model Application #############################################
        #######################################################################################

        #######################################################################################
        ########### Predict the Cluster #######################################################
        ########### Calculate Silhoutte Score #################################################
        #######################################################################################

        X['cluster'] = kmeans_model.fit_predict(X[X.columns])
        print('Silhoutte Score : ' + str(metrics.silhouette_score(X, labels1, metric='euclidean')))
        global score
        score = round((1 + metrics.silhouette_score(X, labels1, metric='euclidean')) / 2 * 100, 2)
        print('Optimal Number of Clusters : ' + str(K[kIdx]))
        global info
        info = X.cluster.value_counts()

        viz1 = pd.concat([viz1, X.cluster], axis=1)
        viz1.to_csv(fname)

        ######## Test #########################################################################

        print ('Accuracy:', metrics.accuracy_score(X.cluster, kmeans_model.labels_))
        print ('Iterations:', kmeans_model.n_iter_)
        print ('Inertia:', kmeans_model.inertia_)
        print ('Elapsed time:', elapsed_rand)

        ######## End Test #####################################################################

        #######################################################################################
        ################ End Prediction and Metrics Calculation ###############################
        #######################################################################################

        #######################################################################################
        ##### Random Forest Classifier for Feature Importance #################################
        #######################################################################################

        p, y = X_imp, X.cluster
        clf = ExtraTreesClassifier()
        clf = clf.fit(p, y)
        fea_imp = pd.DataFrame(clf.feature_importances_)
        fea_imp2 = pd.concat([z1, fea_imp], axis=1)
        fea_imp3 = fea_imp2.values
        fea_imp4 = fea_imp3[fea_imp3[:, 1].argsort()[::-1]]

        p2, y2 = X_imp2, X.cluster
        clf2 = ExtraTreesClassifier()
        clf2 = clf2.fit(p2, y2)
        fea2_imp = pd.DataFrame(clf2.feature_importances_)
        fea2_imp2 = pd.concat([z2, fea2_imp], axis=1)
        fea2_imp3 = fea2_imp2.values
        fea2_imp4 = fea2_imp3[fea2_imp3[:, 1].argsort()[::-1]]

        global feature
        feature = fea2_imp4

        ########################################################################################
        ############ End of Feature Importance #################################################
        ########################################################################################

        ########################################################################################
        ################### Decision Tree for Rules ############################################
        ########################################################################################
        # Initialize model

        tree_model = DecisionTreeClassifier()

        tree_model.fit(X=X_imp, y=X.cluster)

        list1 = []
        list2 = []
        list3 = []

        def get_lineage(tree, feature_names):
            left = tree.tree_.children_left
            right = tree.tree_.children_right
            threshold = tree.tree_.threshold
            features = [feature_names[i] for i in tree.tree_.feature]
            value = tree.tree_.value
            # get ids of child nodes
            idx = np.argwhere(left == -1)[:, 0]

            def recurse(left, right, child, lineage=None):
                if lineage is None:
                    lineage = [child]
                if child in left:
                    parent = np.where(left == child)[0].item()
                    split = 'l'
                else:
                    parent = np.where(right == child)[0].item()
                    split = 'r'

                if split == 'l':
                    lineage.append((str(features[parent]), " <= ", round(threshold[parent], 2)))
                else:
                    lineage.append((str(features[parent]), " > ", round(threshold[parent], 2)))

                if parent == 0:
                    lineage.reverse()
                    return lineage
                else:
                    return recurse(left, right, parent, lineage)

            for child in idx:
                for node in recurse(left, right, child):
                    list2.append(node)

        get_lineage(tree_model, X_imp.columns)

        for i in range(len(list2)):
            if len(str(list2[i])) < 3:
                list3.append("then Cluster = " + str(tree_model.tree_.value[list2[i]].argmax(axis=1)))
            elif len(str(list2[i])) > 3:
                list3.append(str(list2[i]))

        str2 = 'If '
        list4 = []
        p = 0
        for i in range(len(list3)):
            if list3[i].startswith('then'):
                str2 = str2 + list3[i]
                list4.append(str2)
                str2 = 'If '

            else:
                str2 = str2 + list3[i] + ' and '

        list5 = []

        for i in range(len(list4)):
            list4[i] = list4[i].replace("and then", "then")
            list4[i] = list4[i].replace(", \' > \', 0.5", "= 1")
            list4[i] = list4[i].replace(", \' <= \', 0.5", "= 0")
            list5.append({'Sr. No.': range(1, len(list4) + 1)[i], 'Rules': list4[i]})

        rules_final = pd.DataFrame(list5, columns=['Rules'])
        global tab5
        tab5 = rules_final.to_html(classes="w3-table-all table-condensed w3-hoverable w3-small w3-card-8")
        tab5 = tab5.replace("border", 'id="t1" border')

        #######################################################################################
        ########## End of Decision Tree Model and its Rules Derivation ########################
        #######################################################################################

        #######################################################################################
        ########### Decision Tree in Graphical Form ###########################################
        #######################################################################################

        tree.export_graphviz(tree_model, out_file=directory + 'tree_org.dot', feature_names=X_imp.columns, label='root',
                             filled=True, proportion=True, rounded=True)
        check_call(['dot', '-Tpng', directory + 'tree_org.dot', '-o', directory + 'Tree_Output.png'])

        #######################################################################################
        ############## End of Graphical Decision Tree #########################################
        #######################################################################################

        #######################################################################################
        # Joining the original and Processed DataFrame to get Cluster Info. on Original Variables ###
        df = pd.concat([X_imp, X], axis=1)

        result_mean = pd.DataFrame()
        result_mf = pd.DataFrame()
        result_min = pd.DataFrame()
        result_max = pd.DataFrame()
        result = pd.DataFrame(columns=('Features', 'Cluster', 'Mean', 'Min', 'Max'))
        s2 = pd.Series([''])
        j = 1
        x = []
        y = (['', '', '', '', ''])
        z = []
        y2 = (['', '', '', ''])

        ######### Creating No. of Cluster Columns according to the Clusters in Model ############

        for i in range(K[kIdx]):
            df['Cluster_' + str(i)] = df.cluster == i

        #########################################################################################

        #########################################################################################
        ########## Cluster Comparison ###########################################################
        #########################################################################################

        a = []
        c = []
        d = (['', '', '', '', ''])

        for i in range(K[kIdx]):
            result_mean = df.groupby('Cluster_' + str(i))[
                [fea_imp4[0][0], fea_imp4[1][0], fea_imp4[2][0], fea_imp4[3][0]]].mean()
            b = (['Cluster ' + str(i) + ' :', '', '', '', ''])
            a.append(b)

            for j in range(2):
                c = ([result_mean.index.values[j], round(result_mean[result_mean.columns.values[0]][j], 2),
                      round(result_mean[result_mean.columns.values[1]][j], 2),
                      round(result_mean[result_mean.columns.values[2]][j], 2),
                      round(result_mean[result_mean.columns.values[3]][j], 2)])
                a.append(c)

            a.append(d)

        result_c = pd.DataFrame(a, columns=('Cluster', result_mean.columns[0], result_mean.columns[1], result_mean.columns[2], result_mean.columns[3]))

        for i in range(K[kIdx]):
            result_c['Cluster'][4 * i] = 'Cluster ' + str(i) + ' :'
            result_c['Cluster'][4 * i + 1] = 'Mean(Other Clusters)'
            result_c['Cluster'][4 * i + 2] = 'Mean(Cluster ' + str(i) + ')'
            result_c['Cluster'][4 * i + 3] = ''
        global tab1
        tab1 = result_c.to_html(classes="w3-table-all w3-hoverable w3-small w3-card-8")
        tab1 = tab1.replace("border", 'id="t1" border')

        ########################## End Comparison Clusters ########################################

        ###########################################################################################
        ########## Cluster Information ############################################################
        ###########################################################################################

        for i in range(K[kIdx]):
            result['Mean'] = df[df.cluster == i][
                [fea_imp4[0][0], fea_imp4[1][0], fea_imp4[2][0], fea_imp4[3][0]]].mean()
            result['Min'] = df[df.cluster == i][[fea_imp4[0][0], fea_imp4[1][0], fea_imp4[2][0], fea_imp4[3][0]]].min()
            result['Max'] = df[df.cluster == i][[fea_imp4[0][0], fea_imp4[1][0], fea_imp4[2][0], fea_imp4[3][0]]].max()
            result['Cluster'] = str(i)
            result['Features'] = result.index.values
            for j in range(4):
                z = ([result.index.values[j], str(i), df[df.cluster == i][[fea_imp4[j][0]]].mean()[0],
                      df[df.cluster == i][[fea_imp4[j][0]]].min()[0], df[df.cluster == i][[fea_imp4[j][0]]].max()[0]])
                x.append(z)

            x.append(y)
        result_final = pd.DataFrame(x, columns=('Features', 'Cluster', 'Mean', 'Min', 'Max'))
        global tab2
        tab2 = result_final.to_html(classes="w3-table-all table-condensed w3-hoverable w3-small w3-card-8")
        tab2 = tab2.replace("border", 'id="t2" border')
        ############### End Cluster Information ##################################################################

        ##########################################################################################################
        ######## PCA Factors Positive and Negative ###############################################################
        ##########################################################################################################

        X1 = X.iloc[:, :-1]
        pca = PCA(n_components=2).fit(X1)
        pca1 = PCA(n_components=3).fit(X1)

        ncomp = 2
        aa = []
        for i in range(ncomp):
            b = "PCA" + str(i + 1)
            j5 = [zd[idx] for idx in range(len(pca.components_[i])) if pca.components_[i][idx] >= 0.2]
            j3 = [zd[idx] for idx in range(len(pca.components_[i])) if pca.components_[i][idx] <= -0.2]
            aa.append({'Factor': b, 'Positive Parameter': j5, 'Negative Parameter': j3})

        result_factor = pd.DataFrame(aa, columns=('Factor', 'Positive Parameter', 'Negative Parameter'))
        global tab3
        tab3 = result_factor.to_html(classes="w3-table-all table-condensed w3-hoverable w3-small w3-card-8")
        tab3 = tab3.replace("border", 'id="t3" border')

        ################################# End PCA Factors ########################################################

        ##########################################################################################################
        ########### Dominant Variables of PCA Dimensions #########################################################
        ##########################################################################################################

        ab = []
        ac = []
        ad = []
        ae = []
        ab = X_imp.columns.values
        ac = np.absolute(pca.components_[0])
        ad = np.absolute(pca.components_[1])
        ae = np.column_stack((ab, ac, ad))
        result_factor2 = pd.DataFrame(ae, columns=('Features', 'PCA Dimension 1', 'PCA Dimension 2'))
        result_factor2_srt1 = result_factor2.sort_values(by='PCA Dimension 1')
        result_factor2_srt2 = result_factor2_srt1[::-1]
        result_factor2_srt3 = result_factor2.sort_values(by='PCA Dimension 2')
        result_factor2_srt4 = result_factor2_srt3[::-1]

        pm = []
        p = "PCA Dimension 1"
        q = [str(result_factor2_srt2['Features'].head(1).values[0]),
             str(result_factor2_srt2['Features'].head(2).tail(1).values[0]),
             str(result_factor2_srt2['Features'].head(3).tail(1).values[0])]
        r = "PCA Dimension 2"
        s = [str(result_factor2_srt4['Features'].head(1).values[0]),
             str(result_factor2_srt4['Features'].head(2).tail(1).values[0]),
             str(result_factor2_srt4['Features'].head(3).tail(1).values[0])]
        pm.append({'Dimension': p, 'Dominant Variable 1 ': q[0], 'Dominant Variable 2 ': q[1], 'Dominant Variable 3 ': q[2]})
        pm.append({'Dimension': r, 'Dominant Variable 1 ': s[0], 'Dominant Variable 2 ': s[1], 'Dominant Variable 3 ': s[2]})

        result_domi = pd.DataFrame(pm, columns=('Dimension', 'Dominant Variable 1 ', 'Dominant Variable 2 ', 'Dominant Variable 3 '))

        global tab4
        tab4 = result_domi.to_html(classes="w3-table-all table-condensed w3-hoverable w3-small w3-card-8")
        tab4 = tab4.replace("border", 'id="t4" border')
        pl.clf()

        ##########################################################################################################
        ############# Cluster 2-D Plot Primary ###################################################################
        ##########################################################################################################

        pca_2d = PCA(n_components=2).fit_transform(X1)
        x_label2 = np.zeros((len(pca_2d), 1), dtype=np.int64)
        for i in range(len(pca_2d)):
            x_label2[i] = X['cluster'][i]

        labels = kmeans_model.fit_predict(pca_2d)
        if len(np.unique(labels1)) == 3:
            for i in range(0, pca_2d.shape[0]):
                if kmeans_model.labels_[i] == 1:
                    c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
                elif kmeans_model.labels_[i] == 0:
                    c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='#FFB6C1', marker='o')
                elif kmeans_model.labels_[i] == 2:
                    c3 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='#32CD32', marker='*')

            pl.xlabel('PCA Dimension 1')
            pl.legend([c1, c2, c3], ['Cluster 1', 'Cluster 0', 'Cluster 2'], loc='upper center',
                      bbox_to_anchor=(0.5, -0.08), ncol=7, prop=fontSmall)
            pl.title('K-means clusters the dataset into 3 clusters')

        elif len(np.unique(labels1)) == 4:
            for i in range(0, pca_2d.shape[0]):
                if kmeans_model.labels_[i] == 1:
                    c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
                elif kmeans_model.labels_[i] == 0:
                    c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='#FFB6C1', marker='o')
                elif kmeans_model.labels_[i] == 2:
                    c3 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='#32CD32', marker='*')
                elif kmeans_model.labels_[i] == 3:
                    c4 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='c', marker='<')

            pl.xlabel('PCA Dimension 1')
            pl.legend([c1, c2, c3, c4], ['Cluster 1', 'Cluster 0', 'Cluster 2', 'Cluster 3'], loc='upper center',
                      bbox_to_anchor=(0.5, -0.05), ncol=7, prop=fontSmall)
            pl.title('K-means clusters the dataset into 4 clusters')

        elif len(np.unique(labels1)) == 5:
            for i in range(0, pca_2d.shape[0]):
                if kmeans_model.labels_[i] == 1:
                    c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
                elif kmeans_model.labels_[i] == 0:
                    c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='#FFB6C1', marker='o')
                elif kmeans_model.labels_[i] == 2:
                    c3 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='#32CD32', marker='*')
                elif kmeans_model.labels_[i] == 3:
                    c4 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='c', marker='<')
                elif kmeans_model.labels_[i] == 4:
                    c5 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='^')

            pl.xlabel('PCA Dimension 1')
            pl.legend([c1, c2, c3, c4, c5], ['Cluster 1', 'Cluster 0', 'Cluster 2', 'Cluster 3', 'Cluster 4'],
                      loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=7, prop=fontXSmall)
            pl.title('K-means clusters the dataset into 5 clusters')

        elif len(np.unique(labels1)) == 6:
            for i in range(0, pca_2d.shape[0]):
                if kmeans_model.labels_[i] == 1:
                    c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
                elif kmeans_model.labels_[i] == 0:
                    c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='#FFB6C1', marker='o')
                elif kmeans_model.labels_[i] == 2:
                    c3 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='#32CD32', marker='*')
                elif kmeans_model.labels_[i] == 3:
                    c4 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='c', marker='<')
                elif kmeans_model.labels_[i] == 4:
                    c5 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='^')
                elif kmeans_model.labels_[i] == 5:
                    c6 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='#FFA500', marker='v')

            pl.xlabel('PCA Dimension 1', fontsize=18)
            pl.legend([c1, c2, c3, c4, c5, c6],
                      ['Cluster 1', 'Cluster 0', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'],
                      loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=7, prop=fontXSmall)
            pl.title('K-means clusters the dataset into 6 clusters')

        elif len(np.unique(labels1)) == 7:
            for i in range(0, pca_2d.shape[0]):
                if kmeans_model.labels_[i] == 1:
                    c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
                elif kmeans_model.labels_[i] == 0:
                    c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='#FFB6C1', marker='o')
                elif kmeans_model.labels_[i] == 2:
                    c3 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='#32CD32', marker='*')
                elif kmeans_model.labels_[i] == 3:
                    c4 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='c', marker='<')
                elif kmeans_model.labels_[i] == 4:
                    c5 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='^')
                elif kmeans_model.labels_[i] == 5:
                    c6 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='#FFA500', marker='v')
                elif kmeans_model.labels_[i] == 6:
                    c7 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='#D2691E', marker='x')

            pl.xlabel('PCA Dimension 1', fontsize=18)
            pl.legend([c1, c2, c3, c4, c5, c6, c7],
                      ['Cluster 1', 'Cluster 0', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6'],
                      loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=7, prop=fontXXSmall)
            pl.title('K-means clusters the dataset into 7 clusters')

        ax = plt.gca()
        centroids1 = kmeans_model.cluster_centers_
        pl.scatter(centroids1[:, 0], centroids1[:, 1], marker='x', s=50, linewidths=4, color='k', zorder=10)
        radii = [cdist(pca_2d[labels == i], [center]).max() for i, center in enumerate(centroids1)]
#        for c, r in zip(centroids1, radii):
#            ax.add_patch(pl.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

        pl.ylabel('PCA Dimension 2')
        pl.savefig(directory + "KMeans_Primary.png")

        ############################# End Primary Cluster Plot ##########################################

        #################################################################################################
        ############## Cluster 2D Plot Secondary ########################################################
        #################################################################################################

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].
        kmeans_model2.fit(pca_2d)

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = pca_2d[:, 0].min() - 1, pca_2d[:, 0].max() + 1
        y_min, y_max = pca_2d[:, 1].min() - 1, pca_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans_model2.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        plt.plot(pca_2d[:, 0], pca_2d[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans_model2.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
        plt.title('K-means clustering on the given Dataset\n'
                  'Centroids are marked with white cross')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.xlabel('PCA Dimension 1', fontsize=18)
        plt.ylabel('PCA Dimension 2', fontsize=16)
        plt.savefig(directory + "KMeans_Secondary.png")

        ################## End Secondary Plot ##########################################################

        plt.clf()

        def plot_kmeans(kmeans, X, n_clusters=K[kIdx], rseed=0, ax=None):
            labels = kmeans.fit_predict(X)

            # plot the input data
            ax = ax or plt.gca()
            ax.axis('equal')
            for i in range(0, pca_2d.shape[0]):
                if kmeans.labels_[i] == 0:
                    ax.scatter(X[i, 0], X[i, 1], c='r', s=20, zorder=2, marker='o')
                elif kmeans.labels_[i] == 1:
                    ax.scatter(X[i, 0], X[i, 1], c='#FFB6C1', s=20, zorder=2, marker='+')
                elif kmeans.labels_[i] == 2:
                    ax.scatter(X[i, 0], X[i, 1], c='#32CD32', s=20, zorder=2, marker='^')
                elif kmeans.labels_[i] == 3:
                    ax.scatter(X[i, 0], X[i, 1], c='c', s=20, zorder=2, marker='x')
                elif kmeans.labels_[i] == 4:
                    ax.scatter(X[i, 0], X[i, 1], c='b', s=20, zorder=2, marker='v')
                elif kmeans.labels_[i] == 5:
                    ax.scatter(X[i, 0], X[i, 1], c='#FFA500', s=20, zorder=2, marker='*')
                elif kmeans.labels_[i] == 6:
                    ax.scatter(X[i, 0], X[i, 1], c='#D2691E', s=20, zorder=2, marker='<')

            # plot the representation of the KMeans model
            centers = kmeans.cluster_centers_
            radii = [cdist(X[labels == i], [center]).max()
                     for i, center in enumerate(centers)]
            for c, r in zip(centers, radii):
                ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=K[kIdx], alpha=0.5, zorder=1))
            plt.xlabel('PCA Dimension 1')
            plt.ylabel('PCA Dimension 2')
            plt.savefig(directory + "KMeans_Tertiary.png")

        plot_kmeans(kmeans_model, pca_2d)

        return render(request, 'application/process.html')


def show_table(request):
    if not request.user.is_authenticated():
        return render(request, 'application/login.html')
    else:
        return render(request, 'application/tables.html', {"tab1": tab1, "tab2": tab2, "tab3": tab3, "tab4": tab4})


def show_chart(request):
    if not request.user.is_authenticated():
        return render(request, 'application/login.html')
    else:
        directory = 'application/images/user_' + str(request.user.id) + '/result/'
        chart1 = directory + 'KMeans_Primary.png'
        chart2 = directory + 'KMeans_Secondary.png'
        chart3 = directory + 'KMeans_Tertiary.png'

        return render(request, 'application/charts.html',
                      {'chart1': chart1, 'chart2': chart2, 'chart3': chart3, "tab4": tab4})


def decision_tree(request):
    if not request.user.is_authenticated():
        return render(request, 'application/login.html')
    else:
        directory = 'application/images/user_' + str(request.user.id) + '/result/'
        figure = directory + 'Tree_Output.png'
        return render(request, 'application/tree.html', {'table': tab5, 'fig': figure, 'conversion': D})


def fscharts(request):
    if not request.user.is_authenticated():
        return render(request, 'application/login.html')
    else:
        dataSource = {}
        dataSource['chart'] = {
            "caption": "Cluster Information",
            "startingangle": "120",
            "showlabels": "0",
            "showlegend": "1",
            "enablemultislicing": "0",
            "slicingdistance": "15",
            "showpercentvalues": "1",
            "showpercentintooltip": "0",
            "plottooltext": "$label Number of samples : $datavalue",
            "theme": "ocean"
        }

        dataSource['data'] = []
        for i in range(len(info)):
            data = {}
            data['label'] = "Cluster : " + str(info.keys()[i].item())
            data['value'] = info.values[i].item()
            dataSource['data'].append(data)

        pie3d = FusionCharts("pie3d", "ex2", "100%", "400", "chart-1", "json", dataSource)
        # The data is passed as a string in the `dataSource` as parameter.

        dataSource1 = {}
        dataSource1["chart"] = {
            "caption": "Clustering Score",
            "subcaption": "In Percentage",
            "lowerlimit": "0",
            "upperlimit": "100",
            "lowerlimitdisplay": "Bad",
            "upperlimitdisplay": "Good",
            "numbersuffix": "%",
            "tickvaluedistance": "10",
            "gaugeinnerradius": "0",
            "bgcolor": "FFFFFF",
            "pivotfillcolor": "333333",
            "pivotradius": "8",
            "pivotfillmix": "333333, 333333",
            "pivotfilltype": "radial",
            "pivotfillratio": "0,100",
            "showtickvalues": "1",
            "majorTMThickness": "2",
            "majorTMHeight": "15",
            "minorTMHeight": "3",
            "showborder": "0",
            "plottooltext": "<div>Clustering Score : <b>$value%</b></div>",
        }
        dataSource1["colorrange"] = {
            "color": [{
                "minvalue": "0",
                "maxvalue": "30",
                "code": "e44a00"
            }, {
                "minvalue": "30",
                "maxvalue": "70",
                "code": "f8bd19"
            }, {
                "minvalue": "70",
                "maxvalue": "100",
                "code": "6baa01"
            }]
        }
        dataSource1["dials"] = {
            "dial": [{
                "value": str(score),
                "rearextension": "15",
                "radius": "100",
                "bgcolor": "333333",
                "bordercolor": "333333",
                "basewidth": "8"
            }]
        }
        angularGauge = FusionCharts("angulargauge", "ex1", "450", "270", "chart-2", "json", dataSource1)

        return render(request, 'application/fscharts.html',
                      {'output': pie3d.render(), 'output1': angularGauge.render()})


def boxplot(request):
    if not request.user.is_authenticated():
        return render(request, 'application/login.html')
    else:
        dataSource2 = {}
        boxplt = FusionCharts('boxandwhisker2d', 'chartobject-1', '100%', '500', 'chart-1', 'json', dataSource2)
        return render(request, 'application/boxplot.html', {'output': boxplt.render(), 'numericals': num_var})


def boxplot_param(request, sep):
    if not request.user.is_authenticated():
        return render(request, 'application/login.html')
    else:
        val = viz1.cluster.unique()
        val = sorted(val)
        dataSource2 = {}

        dataSource2["chart"] = {
            "caption": "Boundaries of Clusters",
            "subCaption": "By " + str(sep),
            "paletteColors": "#0075c2,#1aaf5d",
            "bgColor": "#ffffff",
            "captionFontSize": "14",
            "subcaptionFontSize": "14",
            "subcaptionFontBold": "0",
            "showBorder": "0",
            "showCanvasBorder": "0",
            "showAlternateHGridColor": "0",
            "legendBorderAlpha": "0",
            "legendShadow": "0",
            "legendPosition": "right",
            "showValues": "0",
            "showMean": "1",
            "toolTipColor": "#ffffff",
            "toolTipBorderThickness": "0",
            "toolTipBgColor": "#000000",
            "toolTipBgAlpha": "80",
            "toolTipBorderRadius": "2",
            "toolTipPadding": "5"
        }

        dataSource2['categories'] = []
        data1 = {}
        data1['category'] = []
        for i in val:
            data = {}
            data['label'] = "Cluster " + str(i)
            data1['category'].append(data)

        dataSource2['categories'].append(data1)

        dataSource2["dataset"] = [{
            "seriesname": str(sep),
            "lowerBoxColor": "#0075c2",
            "upperBoxColor": "#1aaf5d",
            "data": []
        }]

        data1 = {}
        data1['data'] = []
        for i in val:
            data = {}
            abc = viz1[sep][viz1['cluster'] == i].to_json(orient='values')
            abc = abc.replace("[", "")
            abc = abc.replace("]", "")
            data['value'] = abc
            data1['data'].append(data)
        dataSource2["dataset"].append(data1)
        boxplt = FusionCharts('boxandwhisker2d', 'chartobject-1', '100%', '500', 'chart-1', 'json', dataSource2)

        return render(request, 'application/boxplot.html', {'output': boxplt.render(), 'numericals': num_var})


def features_selection(request):
    if not request.user.is_authenticated():
        return render(request, 'application/login.html')
    else:
        dataSource = {}
        dataSource['chart'] = {
            "caption": "Feature Importance",
            "subCaption": "In Percentage",
            "xAxisName": "Features",
            "yAxisName": "Importance(%)",
            "numberSuffix": "%",
            "paletteColors": "#0075c2",
            "bgColor": "#ffffff",
            "borderAlpha": "0",
            "canvasBorderAlpha": "0",
            "usePlotGradientColor": "0",
            "plotBorderAlpha": "10",
            "placevaluesInside": "1",
            "rotatevalues": "1",
            "valueFontColor": "#ffffff",
            "showXAxisLine": "1",
            "xAxisLineColor": "#999999",
            "divlineColor": "#999999",
            "divLineIsDashed": "1",
            "showAlternateHGridColor": "0",
            "subcaptionFontBold": "0",
            "subcaptionFontSize": "14"
        }

        dataSource['data'] = []

        for i in range(len(feature)):
            data = {}
            data['label'] = str(feature.item(i, 0))
            data['value'] = str(round(feature.item(i, 1) * 100, 2))
            dataSource['data'].append(data)

        # Create an object for the Column 2D chart using the FusionCharts class constructor
        column2D = FusionCharts("column2D", "ex1", "100%", "500", "chart-1", "json", dataSource)
        return render(request, 'application/features.html', {'output': column2D.render()})


def multiplot(request):
    if not request.user.is_authenticated():
        return render(request, 'application/login.html')
    else:
        dataSource = {}
        mscol2D = FusionCharts("mscolumn2d", "ex1", "100%", "500", "chart-1", "json", dataSource)
        return render(request, 'application/multiplot.html', {'output': mscol2D.render(), 'categoricals': cat_var})


def multiplot_param(request, cat):
    if not request.user.is_authenticated():
        return render(request, 'application/login.html')
    else:
        val = viz1.cluster.unique()
        val = sorted(val)
        dataSource = {}

        # Chart data is passed to the `dataSource` parameter, as hashes, in the form of key-value pairs.
        dataSource['chart'] = {
            "caption": "Distribution of Clusters",
            "subCaption": "By " + str(cat),
            "xAxisname": "Clusters",
            "yAxisName": "Number of Instances",
            "theme": "ocean"
        }

        dataSource['categories'] = []
        data1 = {}
        data1['category'] = []
        for i in val:
            data = {}
            data['label'] = "Cluster " + str(i)
            data1['category'].append(data)

        dataSource['categories'].append(data1)

        dataSource["dataset"] = []
        uniq = viz1[cat].unique()
        for s in uniq:
            data1 = {}
            data1['data'] = []
            data1['seriesname'] = str(s)
            for i in val:
                data = {}
                abc = len(viz1[cat][(viz1['cluster'] == i) & (viz1[cat] == s)])
                data['value'] = str(abc)
                data1['data'].append(data)
            dataSource["dataset"].append(data1)

        # Create an object for the Multiseries column 2D charts using the FusionCharts class constructor
        mscol2D = FusionCharts("mscolumn2d", "ex1", "100%", "500", "chart-1", "json", dataSource)

        return render(request, 'application/multiplot.html', {'output': mscol2D.render(), 'categoricals': cat_var})
