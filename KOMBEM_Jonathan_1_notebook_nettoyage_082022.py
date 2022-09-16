#!/usr/bin/env python
# coding: utf-8

# # Préparation des données

# ### Données obenues auprès de https://world.openfoodfacts.org
# 
# 
# ### Objectif du présent document
# 
# Préparer le jeu de données en en vue de la réalisation d'une analyse univariée, bivariée et exploratoire des variables pertinentes au regard de notre proposition d'application dans le cadre d'un appel d'offre de "Santé Publique".
# 
# 
# ### Contexte de réalisation de l'étude
# 
# Notre étude du jeu de données devra être simple à comprendre pour un public néophyte. 
# Nous devrons donc être particulièrement attentifs à la lisibilité et aux choix des graphiques pour illustrer notre propos.
#     
# ### Direction de l'étude
# 
# Après une analyse univariée et bivariée du jeu de données, nous pourrons réaliser une ACP afin de comprendre les caractéristiques les plus discriminantes de notre ensemble de produits, ainsi qu'un partitionnement afin de plus simplement 
# catégoriser les produits (beaucoups de catégories présentes dans le jeu).
# 
# Aussi nous pourrons effectuer quelques régressions pour étudier par exemple la correspondance entre nutriscore et nombre d'ingrédients dans un produit, la présence d'additifs ou d'allergènes.
# 
# Enfin, nous proposerons une métrique rendant compte de la qualité intrinsèque des produits, ainsi que de leur impact sur l'environnement, une métrique qui se voudra plus transparente envers le consommateur sur le bien fondé de la proposition de valeur qu'ils représentent.
# 
# ### Idée d'application
# 
# Nous proposons un outil permettant au consommateur de mieux apprécier la qualité des produits qu'il consomme, et qui sera principalement basé sur l'appréciation des critères suivants :
# - Le nustricore (valeur numérique)
# - Le critère NOVA qui range en 4 catégories les produits en fonction de leur taux de transformation industrielle
# - La présence d'addifits, d'allergènes, d'ingrédidents issus de l'huile de palme
# - La disparité entre l'origine du produit, son lieu de transformation et la FRANCE
# - L'appartenance du produit à un label BIO ou qui s'engage à prendre mieux soin des consommateurs
# - Le type de packaging utilisé pour conditionner le produit

# ## Récupération des données

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import missingno as msno
import FETCH_LOAD_DATAS


# In[2]:


help(FETCH_LOAD_DATAS)


# ## Découverte du jeu de données

# In[3]:


df_food = FETCH_LOAD_DATAS.load_food_data()


# In[4]:


df = df_food.copy()


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df.isna().mean().mean()


# In[8]:


#trier par valeur de taux de remplissage
msno.bar(df, sort='ascending')


# In[9]:


#cellule utilisée pour naviguer à traver les colonnes pour se faire une idée des valeurs qu'elles contiennent
df['traces_tags'].value_counts()[20:]


# In[10]:


#cellule utilisée pour naviguer dans le jeu de données pour observer les taux de remplissage par bloc
msno.matrix(df.iloc[:1000,58:])


# In[11]:


df.isna().mean(axis=1).hist(bins=100)


# In[12]:


df.isna().mean(axis=1)[df.isna().mean(axis=1) > 0.75]


# In[13]:


df.isna().mean().hist(bins=100)


# In[14]:


df.isna().mean()[df.isna().mean() > 0.90]


# ### Description du jeu de données
# 
# Le jeu de données est un ensemble de 320772 lignes et 160 colonnes qui présente un certain nombre de caractéristiques de produits alimentaires : 
# - <b>générales</b> (origine, site de transformation, packaging, ingrédients, etc.)
# - <b>particulières</b> (présence d'additifs, présence d'allergènes, appartenance à un label bio ou autre, etc.)
# - <b>nutritionnelles</b> (taux de protéines, fibres, graisses monosaturées, vitamine b6, etc.)
# 
# Les caractéristiques de chaque produit sont réparties en <u>5 parties significatives</u> :
# 1. Une partie générale relative à la base de données du site https://world.openfoodfacts.org/data
# 2. Une partie informative contenant des métadonnées du produit (packaging, origine, lieu de transformation, etc.)
# 3. Une partie 'constitution du produit' avec les ingrédients, allergènes et traces d'autres produits qu'il peut contenir
# 4. Une partie présentant des informations diverses telles que la présence d'additifs, d'huile de palme et notamment sur le NUTRISCORE** du produit
# 5. Une partie nutritionnelle où l'on retrouve toutes les informations typiques d'énergie pour 100g de produit consommé, la teneur en 96 nutriments différents pour 100 g de produits ainsi que le nutriscore** du produit.
# 
# ** On fait ici la différence entre le NUTRISCORE (A,B,C,D,E) et le nutriscore qui est la valeur numérique avant transformation en catégorie (A,B,C,D,E). Cette distinction sera faite dans toute la suite de notre étude.
# 
# Le jeu de données présente un taux de valeurs manquantes de <b>76%</b> :
# - Près de la moitié des colonnes présentent un taux de valeurs manquantes supérieur à <b>99%</b>, mode de la distribution des  taux de valeurs manquantes des colonnes (74 colonnes)
# - Plus de 60% des colonnes présentent un taux de valeurs manquantes supérieur à <b>90%</b> (100 colonnes)
# - Plus de 2/3 des lignes (71%) présentent un taux de valeurs manquantes supérieur à <b>75%</b>, mode de la distribution des taux de valeurs manquantes des lignes (229030 lignes)

# ## Sélection des variables utiles
# 
# Pour la suite de notre étude du jeu de données, nous allons, partie par partie décortiquer les variables que nous allons garder, et selon ces colonnes, les individus dont nous allons nous séparer.
# 
# Comme il y a significativement plus valeurs pour les colonnes du début du tableau, que pour les colonnes du milieu vers la fin du tableau (à l'exception du nutriscore), nous considérons que nous pourrons épurer celui-ci de gauche à droite, de partie en partie, à quelques exceptions près de variables peut-être.

# #### 1. Partie générale relative à la base de données des produits

# In[15]:


col1 = list(df.columns[:df.columns.get_loc('packaging')])
col1


# Nous n'allons conserver que les colonnes 'code', 'url', 'last_modified_datetime' et 'product_name'.
# - Nous espérons à ce stade utiliser la colonne 'code' comme identifiant unique pour chaque produit.
# - La colonne 'url' renvoie à la page produit sur le site "openfoodfacts"
# - La colonne 'last_modified_datetime' pourra nous prévenir de l'ajout d'information sur un produit que nous pourrons alors peut être mieux analyser
# - La colonne 'product_name' pour permettre à l'utilisateur de notre application de réaliser une recherche textuelle simple
# 

# In[16]:


col_part1_to_drop = [x for x in col1 if x not in ['code','url','last_modified_datetime','product_name' ]]
df.drop(columns=col_part1_to_drop, inplace=True)


# In[17]:


df.dropna(subset=['code'],inplace=True, axis=0)


# In[18]:


df[df['code'].duplicated(keep=False)].sort_values('code', ascending=True).head()


# Pour les doublons, à ce stade, nous n'allons garder que l'occurence présentant le plus faible taux de valeurs manquantes.

# In[19]:


df['taux_Nan'] = df.isna().mean(axis=1)
df.sort_values('taux_Nan', ascending=False, inplace=True)
df.drop_duplicates(subset=['code'], keep='first', inplace=True)
df.drop('taux_Nan', axis=1, inplace=True)


# In[20]:


df.shape


# In[21]:


df['code'].isna().mean()


# In[22]:


df['code'].nunique()/df['code'].shape[0]


# In[23]:


df_1 = df.copy()


# In[24]:


df['url'].str.startswith('http').sum()


# La colonne 'code' peut désormais être considérée comme notre colonne identifiante pour nos données.
# Le travail sur le jeu de données vis-à-vis de la partie 1 est terminée.

# ### RAF partie 1 :
# - vérifier la validité des 'url' (https://www.moonbooks.org/Articles/Vérifier-si-une-adresse-url-existe-avec-python/)
# - vérifier la validité des dates de la colonne 'last_modified_datetime'

# #### 2. Partie informative contenant des métadonnées du produit

# In[25]:


df = df_1.copy()


# In[26]:


col2 = list(df.columns[df.columns.get_loc('packaging'):(df.columns.get_loc('countries_fr')+1)])


# In[27]:


df[col2].isna().mean()


# In[28]:


# cellule utilisée pour explorer les différents nombres de valeurs uniques
df['categories_fr'].value_counts()


# - Dans un premier temps, nous n'allons garder que les produits distribués en France en effectuant un tri sur la colonne 'countries_fr', puis nous allons supprimer toutes les colonnes 'countries_x'
# 
# 
# - Ensuite, nous allons supprimer toutes les colonnes "_tags" car elles comportent les mêmes données que les colonnes auxquelles elles sont associées, mais avec une mise en forme qui les rend plus difficilement lisibles.
#     
# 
# - Nous allons également supprimer les colonnes 'cities', 'cities_tags' ainsi que 'purchase_places' car nous considérons que les produits sont disponibles partout en France, ainsi que la colonne 'categories' qui est redondante avec la colonne 'categories_fr' et moins synthétique (moins de valeurs uniques pour un taux de remplissage identique) et pour les mêmes raisons, nous allons supprimer les colonnes 'emb_codes' et 'first_packaging_code_geo' qui nous renseignent sur la ville dans laquelle le produit a été conditionné (ville française pour l'information dans 'first_packaging_code_geo', pas nécessairement en France pour l'information dans 'emb_codes', <i><u>mais l'on suppose ici qu'un produit emballé à l'étranger, vendu en France, a été transformé à l'étranger</i></u>**, et nous avons donc l'information dans la colonne 'manufacturing_places').
# 
# ** Nous espérons vraiment que c'est le cas...

# In[29]:


df = df[df['countries_fr'].str.contains('France', regex=False).fillna(False)]


# In[30]:


L_split = [col.split('_') for col in col2]
L_tags = ['tags' in x for x in L_split]
col_part2_to_drop = [x for x,y in zip(col2, L_tags) if y]
col_part2_to_drop.append('countries')
col_part2_to_drop.append('countries_fr')
col_part2_to_drop.append('purchase_places')
col_part2_to_drop.append('cities')
col_part2_to_drop.append('categories')
col_part2_to_drop.append('emb_codes')
col_part2_to_drop.append('first_packaging_code_geo')
df.drop(columns=col_part2_to_drop, inplace=True)


# <b>Analysons la colonne 'packaging' :</b>

# In[31]:


# pour faciliter le travail sur les chaînes de caractères, on les passe toutes en minuscule
df['packaging'] = df['packaging'].str.lower()


# In[32]:


df['packaging'].value_counts()


# In[33]:


#cellule utilisée pour naviguer à traver les diffférentes valeurs uniques d'emballage
df['packaging'].value_counts()[50:70]


# In[34]:


#fonction retournant la liste contenant pour chaque valeur unique d'une colonne, la liste des sous-éléments qui la composent
def get_list_splits_str(col_name):
    return df[col_name].value_counts().index.str.split(',').tolist()

# fonction permettant de retourner la liste des sous-éléments uniques contenue dans une colonne
def get_list_uniques_splits_str(col_name):
    list_tmp = []
    list_splits = get_list_splits_str(col_name)
    for i in range(len(list_splits)):
        element = list_splits[i]
        for j in range(len(element)):
            if element[j] not in list_tmp:
                list_tmp.append(element[j])
    return list_tmp


# In[35]:


L_pack_uniques = get_list_uniques_splits_str('packaging')
len(L_pack_uniques)


# In[36]:


L_pack_uniques


# In[37]:


# cellule pour explorer le nombre d'occurence de mots-clés
df['packaging'].str.contains('brique').value_counts()


# In[38]:


# cellule pour explorer les différentes catégories d'emballage contenant un mot-clé
df['packaging'][df['packaging'].str.contains('karton').fillna(False)].value_counts()[:10]


# In[39]:


# cellule pour explorer les noms des produits contenant un mot-clé dans le descriptif de leur emballage
# pour mieux en appréhender le matériau
df[df['packaging']=='boîte']['product_name']


# In[40]:


L_carton = ['carto', 'papie', 'paper', 'bri', 'tetra', 'cartón', 'tétra', 'doypack', 'briquette', 'boîte à œufs', 'karton',
            'cellulose', '21', 'caton', 'papel', 'cartão', 'craton', 'cartion', 'doyapck', 'wellpappe', 'carrton',
            'boîte à oeufs', 'cardboard']

set_plastique = set(['plast', 'film', 'paquet', 'tetra', 'vide', 'protect', 'tétra', 'pet', 'doypack', 'bac', 'cellophane',
                     'blister', 'fraîcheur', 'plásti', 'sachet','pp5', 'polyprop', 'пластиковый', 'sélophane', 
                     'polyéthylène', 'plstique', 'pebd', 'poliprop', 'pvc', 'ldpe', 'pet', 'pp5', 'polystyr' , 'poliestireno',
                     'gaz', 'souple', 'céllophane', 'plastqiue', 'plastiqe', 'aérosol', 'pe-hd', '5-pp', 'pp-5', 'plastc',
                     'pp 5', '5 pp', 'pastic', 'doyapck', 'pete 1', 'pp', '5 opp','zellophan', 'atmos', 'kunststoff','filet',
                     'plasitque', 'platique', 'ficellle', 'barquette','hdpe', 'pastique', 'palstique', 'plasique', 
                     'plaqtique', 'plasitque'])

L_metal = ['alu', 'métal', 'metal', 'acier', 'conserve', 'tetra', 'tétra', 'tin', 'fût', 'can', 'konserve', 'blister',
           'fer', '40 fe', 'aérosol', 'torebki foliowej','bidon', 'bombe', 'allu', 'alimunium']

L_verre = ['verre', 'bocal', 'glas', 'glass', 'vidrio', 'vetro', 'glaß', 'verrre','vidro', 'szklana']

set_non_recyclable = set(['jeter', 'sulfurisé', 'cellophane', 'blister', 'cuisson', 'ldpe', 'non recyclable', 'céllophane',
                          'zellophan', 'filet', 'ficelle','barquette'])

L_recyclable = ['bois', 'recycle', 'recycla', 'tetra', 'tétra', 'cellulose', 'pp5', 'polyprop', 'pulpe', 'compost',
                'biodégra', 'cagette', 'pehd', 'polyéthylène', 'wood', 'pebd', 'polietile', 'pet', 'aérosol', 'pe-hd',
                'pet', 'pp', 'hdpe']

set_pack_recycl = set(['recyclé', 'consigne'])

set_no_pack = set(['sans conditionnement', 'aucun', 'vrac', 'rien', 'sans emballage'])

set_better_pack = set(['consign', 'sans suremballage', 'pefc', 'staitiegeld', 'caution', 'statiegeld', 'réutilisable',
                       'mehrwegpfand'])

set_over_pack =set(['indiv', 'suremballage', 'dose'])

L_recyclable.extend(L_carton)
L_recyclable.extend(L_metal)
L_recyclable.extend(L_verre)

set_carton = set(L_carton)
set_metal = set(L_metal)
set_verre = set(L_verre)
set_recyclable = set(L_recyclable)

dict_pack = {'carton':set_carton, 'plastique':set_plastique, 'metal':set_metal, 'verre':set_verre, 
             'non_recyclable':set_non_recyclable, 'recyclable':set_recyclable, 'pack_recycl':set_pack_recycl, 
             'pas_demballage':set_no_pack, 'emball_intell':set_better_pack, 'suremballage':set_over_pack}


# In[41]:


n = 0
for value in dict_pack.values():
    n+=len(value)
n - len(L_carton)-len(L_metal)-len(L_verre)


# Nous allons réaliser un tableau disjonctif complet où chaque produit appartiendra à autant de catégories de packaging qu'il ne contient de matières différentes ou porte une mention spécifique ('a jeter', 'recyclable').
# Nous ne pouvons donc pas utiliser la fonction OneHotEncoder de scikit-learn (qui ne peut attribuer qu'une modalité à chaque individu).
# Nous allons créer un ensemble de catégories de matière et mentions spécifiques, et créer une colonne pour chacune des catégories, et nous vérifierons pour chaque produit si son packaging comporte les matières ou mentions spécifiques.
# 
# Le travail préliminaire réalisé ci-avant a permis de passer de 9328 valeurs différentes, à 3251 modalités uniques puis à 156 modalités discriminant 10 catégories d'emballage (division par 61).

# In[42]:


# nous définissons la fonction qui nous indiquera si le packaging d'un produit contient appartient à l'une des catégories 
# définies ci-avant

def belong_pack_catg(pack_cat, value):
    dict_cat = dict_pack[pack_cat]
    if type(value) == float:
        if np.isnan(value):
            n = np.nan
    else:
        for pack in dict_cat:
            contains = str(value).__contains__(pack)
            if contains:
                n = 1
                break
            else:
                n = 0
    return n

# nous définissons ensuite la fonction qui va créer le tableau injonctif des catégories d'emballage
def set_cols_cat_allerg():
    k=1
    func = lambda value: belong_pack_catg(key, value)
    for key in dict_pack.keys():
        df.insert(loc=(df.columns.get_loc('packaging')+k), column=key, value=df['packaging'].apply(func)) 
        k+=1


# In[43]:


set_cols_cat_allerg()


# In[44]:


df[df['packaging'].notna()].loc[:,'carton':'suremballage']


# In[45]:


df[df['packaging'].notna()].loc[:,'packaging'][df[df['packaging'].notna()].loc[:,'carton':'suremballage'].sum(axis=1) == 0].value_counts()[:20]


# In[46]:


(df[df['packaging'].notna()].loc[:,'carton':'suremballage'].sum(axis=1) == 0).value_counts()/df[df['packaging'].notna()].shape[0]


# On s'aperçoit qu'avec notre tableau disjonctif, nous récupérons légèrement plus de 95% de l'information contenue dans la colonne 'packaging', le reste n'étant pas exploitable en l'état sans faire la correspondance avec le nom du produit, qui peut lui-même nous renseigner sur l'emballage utilisé.
# 
# Par ailleurs, grâce à ce dernier travail, nous pourrons facilement extraire de la valeur pour constituer la métrique de notre application.

# Nous n'avons désormais plus besoin de la colonne 'packaging' :

# In[47]:


df.drop('packaging', axis=1, inplace=True)


# <b>Nous allons maintenant travailler sur la colonne 'categories_fr'.</b>
# - Nous allons tâcher de synthétiser les catégories tout en ne perdant pas trop de granularité car nous nous appuierons sur les catégories de produits pour faire des recommandations dans notre application
# - Nous mettrons en évidence la présence de 'viande' et de 'porc' pour les spécificités des certains régimes, la présence de 'poisson', 'fruits de mer', 'oeuf' et 'gluten' sera traitée dans la prochaine partie avec les allergènes.

# In[48]:


plt.figure(figsize=(15,8))
df['categories_fr'].value_counts()[:50].plot.bar()


# In[49]:


# pour faciliter le travail sur les chaînes de caractères, on les passe toutes en minuscule
df['categories_fr'] = df['categories_fr'].str.lower()


# In[50]:


df['categories_fr'].notna().sum()


# In[51]:


df['categories_fr'].nunique()


# In[52]:


# on crée un dictionnaire nous renseignant sur le nombre d'occurences de chaque 'sous-catégorie' 
# parmi les différentes catégories
def dict_subcateg(col_name):
    dict_tmp = {}
    L_split_subcateg = df[col_name].value_counts().index.str.split(',').tolist()
    for liste in L_split_subcateg:
        for element in liste:
            if element not in dict_tmp.keys():
                dict_tmp[element]=1
            else:
                dict_tmp[element]+=1

    dict_subcateg = {}
    sorted_keys = sorted(dict_tmp, key=dict_tmp.get, reverse=True)

    for w in sorted_keys:
        dict_subcateg[w] = dict_tmp[w]

    return dict_subcateg

dict_subcateg('categories_fr')


# Les sous-catégories suivantes sont trop générales :
# - 'aliments et boissons à base de végétaux'
# - "aliments d'origine végétale"
# - 'aliments à base de fruits et de légumes'
# - 'boissons'
# 
# Nous décidons de catégoriser les produits sans elles, en créant une nouvelle colonne en retirant leur présence, et en ne sélectionnant qu'un nombre réduit de catégories pour décrire chaque produit :

# In[53]:


list_cat_to_del = ['aliments et boissons à base de végétaux', "aliments d'origine végétale",
                   'aliments à base de fruits et de légumes', 'boissons']


# In[54]:


# fonction renvoyant la chaîne de caractère 'value' raccourcie dont on a gardé les 'n_synth' premiers éléments 
# séparés par des virgules, en retirant les catégories passées en argument
def synthetize_value(value, n_synth, list_cat_to_del=list_cat_to_del):
    split_value = str(value).split(',')
    for cat in list_cat_to_del:
        split_value = [value for value in split_value if value != cat]
    synth_value = ''
    n = len(split_value)
    if n!=0:
        for i in range(np.min([n,n_synth])-1):
            synth_value += (str(split_value[i])+', ')
        synth_value += str(split_value[(np.min([n,n_synth])-1)])
    return synth_value


# Pour avoir une idée de l'effet de notre catégorisation synthétique, on peut regarder combien de nouvelles modalités différentes il nous faut garder pour décrirer tous nos produits (dans l'optique d'éventuellement réaliser un tableau disjonctif complet ici aussi):

# In[55]:


# on trace pour chaque nombre de sous-catégories que l'on garde pour décrire un produit, l'évolution de la proportion
# de produits que l'on décrit en fonction du nombre de nouvelles modalités que l'on garde
def display_categ_repart(n_synth, n_newmodality):
    L_plot = []
    plt.figure(figsize=(12,6))
    for i in range(n_synth+1):
        L_tmp = []
        Serie_synth_categ = df[df['categories_fr'].notna()].apply(lambda x: synthetize_value(x['categories_fr'],i), axis=1)
        for j in range(1,(n_newmodality+1)):
            L_tmp.append(Serie_synth_categ.value_counts()[:j].sum()/len(Serie_synth_categ))
        plt.plot(list(range(1,(n_newmodality+1))), L_tmp, label=f'n_synth={i}')
   
    plt.legend(bbox_to_anchor=(1,1))
    plt.grid(visible=True)
    plt.show()


# In[56]:


display_categ_repart(3,100)


# On constate sans surprise, qu'en ne gardant qu'une sous-catégorie pour décrire un produit, avec seulement 20 nouvelles modalités, nous décrvions plus de 90% des produits, mais on obtient alors des catégories trop larges pour que la recommandation d'un produit appartenant à la même catégorie puisse toujours être pertinente. Il suffit de comparer les modalités les plus présentes dans le jeu de données pour 'n_synth' = 1, 2 puis 3 pour s'en apercevoir :

# In[57]:


# fonction créant la colonne catégorielle synthétique associée à la valeur de n_synth et contenant les n_synth premières
# sous-catégories de la colonne 'categories_fr' du produit
def set_col_categ_synth(n_synth):
    if 'categories_synth' in df.columns:
        df.drop('categories_synth', axis=1, inplace=True)
    Serie_categ_synth = df.apply(lambda x: synthetize_value(x['categories_fr'],n_synth), axis=1)
    df.insert(loc=(df.columns.get_loc('categories_fr')+1), column='categories_synth', value=Serie_categ_synth)


# In[58]:


set_col_categ_synth(1)
df['categories_synth'].value_counts()[:10]


# In[59]:


set_col_categ_synth(2)
df['categories_synth'].value_counts()[:10]


# In[60]:


set_col_categ_synth(3)
df['categories_synth'].value_counts()[:20]


# Pour la suite de l'étude nous allons garder les colonnes obtenues pour n_synth = 2 et 3, ainsi, si nous ne trouvons pas de produits à recommander dans la catégorie d'un produit associée à n_synth = 3, nous pourrons proposer un élargissement de recommandation à la catégorie de ce produit associée à n_synth = 2 à défaut.

# Pour la suite de l'étude nous n'allons non pas créer un tabelau disjonctif complet, mais créer 3 colonnes catégorielles chacune comportant la valeur de la sous-catégorie de niveau 0, 1 et 2 (ie. n_synth = 1, 2 et 3). Il sera alors facile de proposer des produits similaires à leur recherche nos utilisateurs finaux, et à défaut, de proposer un élargissement de recherche en proposant des produits de la même catégorie parent, ainsi de suite.. 
# 
# Par ailleurs, il sera possible d'encoder chacune des colonnes pour faciliter le traitement les traitements numériques de notre jeu de données ultérieurement !
# 
# Pour cela, nous allons modifier nos fonctions 'synthetize_value' et 'set_col_categ_synth' :

# In[61]:


# fonction retournant une liste des n_synth_max premières catégories d'un produit, en mettant la valeur 'x' si le produit
# n'est pas décrit par n_synth_max catégories
def list_synth_value(value, n_synth_max, list_cat_to_del=list_cat_to_del):
    split_value = str(value).split(',')
    for cat in list_cat_to_del:
        split_value = [value for value in split_value if value != cat]
    list_synth = []
    n=len(split_value)
    for i in range(n_synth_max):
        if i < n:
            if split_value[i] == 'nan':
                list_synth.append(np.nan)
            else:
                list_synth.append(split_value[i])
        else:
            list_synth.append(np.nan)
    return list_synth


# In[62]:


list_synth_value(df.loc[df[df["categories_fr"].notna()].head(1).index[0]]['categories_fr'],3)


# In[63]:


list_synth_value(df.loc[df[df["categories_fr"].isna()].head(1).index[0]]['categories_fr'],3)


# In[64]:


# fonction créant les colonnes catégorielles hiérarchiques des produits, de la catégorie la plus générale (n_synth=1) à la 
# catégorie la plus particulière (n_synth=n_synth_max)
def set_cols_categ_synth(n_synth_max):
    for i in range (n_synth_max):
        col_name = 'categories_synth{}'.format((i+1))
        if col_name in df.columns:
            df.drop(col_name, axis=1, inplace=True)
        Serie_categ_synth = df.apply(lambda x: list_synth_value(x['categories_fr'],n_synth_max)[i], axis=1)
        df.insert(loc=(df.columns.get_loc('categories_fr')+(i+1)), column=col_name, value=Serie_categ_synth)


# In[65]:


set_cols_categ_synth(3)


# In[66]:


df[df['categories_fr'].notna()].iloc[:, df.columns.get_loc('categories_fr'):df.columns.get_loc('categories_synth3')]


# Créons maintenant les colonnes nous permettant de savoir si un produit contient de la viande (pour les végétariens, végétaliens, et flexitariens) et du porc (pour ceux qui n'en consomment pas quelque soit la raison).

# In[67]:


df.insert(loc=(df.columns.get_loc('categories_synth3')+1), column='viande', value=df['categories_fr'].apply(lambda x: 1 if str(x).__contains__('viande') else (0 if str(x) != 'nan' else np.nan) ))
df.insert(loc=(df.columns.get_loc('categories_synth3')+2), column='porc', value=df['categories_fr'].apply(lambda x: 1 if str(x).__contains__('porc') else (0 if str(x) != 'nan' else np.nan) ))


# In[68]:


df[df['porc']==1].loc[df['viande']==0]


# In[69]:


df.loc[df['porc']==1,'viande']=1


# In[70]:


df.loc[df['product_name'].str.contains('Gélatine').fillna(False), 'viande']=0


# In[71]:


df[df['porc']==1].loc[df['viande']==0]


# Nous n'avons dès lors plus besoin de la colonne 'categories_fr' ni 'categories_synth'.

# In[72]:


df.drop(columns=['categories_fr','categories_synth'], axis=1, inplace=True)


# <b>Intéressons nous maintenant aux colonnes 'origins' et 'manufacturing_places'.</b>

# Le but ici sera d'associer à chaque localisation un continent, et nous pourrons ainsi donner un poids (négatif) à chaque continent en fonction de sa "distance" avec la France, pour prendre en compte l'impact environnemental dûe à la distance de transport.
# 
# Pour simplifier notre analyse, nous faisons ici l'hypothèse que <u>l'impact environnemental des moyens de transports des produits ne dépend que de la 'distance'</u>.

# In[73]:


df['origins'] = df['origins'].str.lower()
df['manufacturing_places'] = df['manufacturing_places'].str.lower()


# In[74]:


df['origins'].isna().mean(), df['manufacturing_places'].isna().mean()


# In[75]:


df['origins'].value_counts()


# In[76]:


df['manufacturing_places'].value_counts()


# In[77]:


# on regroupe toutes les localisations contenant 'france' dans 'france' pour dégrossir 
func = lambda x : 'france' if str(x).__contains__('france') else x
df['origins'] = df['origins'].apply(func)
df['manufacturing_places'] = df['manufacturing_places'].apply(func)


# In[78]:


df['origins'][df['origins'].str.contains('france').fillna(False)].value_counts()


# In[79]:


df['manufacturing_places'][df['manufacturing_places'].str.contains('france').fillna(False)].value_counts()


# In[80]:


# on charge un dataframe contenant les correspondances pays-continent en français dont nous avons besoin
df_continents = pd.read_csv('extradatas\\Liste_pays_continents.csv', delimiter=';', on_bad_lines='skip')
df_continents.head()


# In[81]:


df_continents.iloc[:,0] = df_continents.iloc[:,0].str.lower()
df_continents.iloc[:,1] = df_continents.iloc[:,1].str.lower()


# In[82]:


df_continents['Continent'].value_counts()


# In[83]:


# on ne souhaite que 5 continents principaux, donc on regroupe les amériques, et on renomme la colonne 'Nom français'
# pour l'appeler de manière plus intuitive
df_continents['Continent'] = df_continents['Continent'].apply(lambda x: 'amérique' if x.__contains__('amérique') else x)
df_continents = df_continents.rename(columns={'Nom français':'Pays'})


# In[84]:


# on crée la liste L_pays pour faciliter l'exploration des pays, plus simple avec une liste qu'un objet Series
L_pays = [element for element in df_continents['Pays']]
L_pays


# In[85]:


# on simplifie les noms de pays dans notre dataframe pays-continent pour faciliter la correspondance avec notre jeu de données
func = lambda x : x.split('(')[0].strip()
df_continents['Pays'] = df_continents['Pays'].apply(func)


# In[86]:


L_pays = [x.split('(')[0].strip() for x in L_pays]
L_pays


# A ce stade, pour accélérer notre travail, nous allons remplacer chaque localisation dans nos colonnes 'origins' et 'manufacturing_places' par le pays qu'elle contient, et qui fait partie de la colonne 'Pays' de notre dataframe df_continent car alors on pourra lui associer un continent :

# In[87]:


df_continents['Pays'][:20]


# In[88]:


df['origins'].isna().mean()


# In[89]:


# nous définissons la fonction qui nous donnera le pays présent dans une localisation, s'il fait partie du dataframe
# df_continent, sinon la localisation et nous devrons travailler sur sa mise en forme plus manuellement

def belong_to_country(value):
    country = ''
    for element in df_continents['Pays']:
        contains = str(value).__contains__(element)
        if contains:
            country=element
            break
        else:
            country=value
    return country


# In[90]:


df['origins'] = df['origins'].apply(belong_to_country)
df['origins'].value_counts()


# In[91]:


df['manufacturing_places'] = df['manufacturing_places'].apply(belong_to_country)
df['manufacturing_places'].value_counts()


# A ce stade, nous sommes respectivement passés de 1828 à 1153 (-37%) et de 1741 à 1002 (-43%) valeurs uniques de localisation dans nos colonnes 'origins' et 'manufacturing_places', avec un plus grand nombre de valeur en conformité avec notre dataframe <b>df_continents</b>.

# In[92]:


func = lambda x: False if x in df_continents['Pays'].tolist() else True
df['origins'][df['origins'].apply(func)].value_counts()


# A partir d'ici, il nous faut créer un dictionnaire pour mettre en forme nos colonnes 'origins' et 'manufacturing_places' pour que leurs valeurs correspondent à celle de notre colonne 'Pays' dans le dataframe df_continents.
# 
# 
# Par exemple, nous remarquons que ce dernier ne distingue pas les différents pays du royaume-uni et ne contient pas 'équateur...
# De même, certains pays de nos colonnes 'origins' et 'manufacturing_places' ne sont pas en français, ou omettent des accents...
# 
# Nous créons donc un dictionnaire de mise en forme des valeurs des colonnes 'origins' et 'manufacturing_places' vers la forme présente dans la colonne 'Pays' de df_continents, et un dictionnaire pour simplifier cette dernière colonne :

# In[93]:


func = lambda x: False if x in df_continents['Pays'].tolist() else True
df['manufacturing_places'][df['manufacturing_places'].apply(func)].value_counts()


# In[94]:


dict_pays_regroup_df = {'belg':'belgique', 'spa':'espagne','basque':'espagne','pays':'pays-bas', 'ital':'italie', 
                        'royaume':'royaume-uni', 'grande bretagne':'royaume-uni',
                        "grande-bretagne":'royaume-uni', 'angleterre':'royaume-uni', 'portugal':'portugal', 'tha':'thaïlande',
                        'cosse':'royaume-uni', 'taiwan':'taïwan', 'deutschland':'allemagne', 'normandie':'france',
                        'switzerland':'suisse', 'franche':'france', 'unis':'états-unis', 'allemagne':'allemagne',
                        'écosse':'royaume-uni', 'germany':'allemagne', 'uk':'royaume-uni', 'irlande':'royaume-uni', 
                        'bretagne':'france', 'europ':'roumanie', 'agriculture ue':'roumanie', 'quateur':'colombie',
                        'indien':'inde', 'ue / non ue':'roumanie', 'pacifique nord-est':'états-unis', 'india':'inde',
                        'proven':'france', 'antille':'dominicaine', 'kingdom':'royaume-uni', 'alaska':'états-unis',
                        'corse':'france', 'latine':'brésil', 'ue,non ue':'roumanie', 'usa':'états-unis',
                        'agricultura ue,agricultura no ue':'roumanie', 'u.e.':'roumanie', 'china':'chine',
                        'ivoire':"côte d'ivoire", 'savoie':'france', 'ue et non ue':'roumanie', 'fao 34':'sénégal',
                        'champagne':'france', 'nouvelle zélande':'nouvelle-zélande',
                        'agen':'france', 'gironde':'france', 'schweiz':'suisse', 'élaboré en ue':'roumanie', 
                        'vezelay':'france', 'poitou':'france', 'salvetat':'france', 'domingue':'dominicaine', 
                        'aisne':'france','suri':'suriname', 'normandie':'france', 'polska':'pologne',
                        'poland':'pologne','mer du nord':'norvège', 'lorraine':'france', 'auvergne':'france',
                        'ecuador':'mexique', 'alpe':'france', 'atlantique centre est':'sénégal', 
                        'atlantique nord est':'norvège', 'loire':'france', 'atlantique sud-ouest':'brésil',
                        'atlantique n-e':'norvège', 'asie':'chine', 'england':'royaume-uni', 'origine u':'roumanie',
                        'floride':'états-unis', 'riesling':'france', 'palestine':'liban', 'deutchland':'allemagne',
                        'algerie':'algérie', 'pacifique centre-ouest':'indonésie','modène':'italie',
                        'holland':'pays-bas', 'dordogne':'france', 'isigny':'france', 'afrique':'sénégal', 'mornant':'france',
                        'strasbourg':'france', 'manche':'royaume-uni', 'bordeaux':'france', 'gréce':'grèce', 
                        'costa':'costa rica', 'fao 51':'madagascar', 'voges':'france', 'adour':'france', 'roussillon':'france',
                        'rhône':'france', 'guat':'guatemala', 'gascogne':'france', 'south africa':'afrique du sud',
                        'aveyron':'france', 'galmier':'france', 'abbat':'france', 'australi':'australie', 'garonne':'france',
                        'ducey':'france', 'royaume-uni':'royaume-uni', 'gascogne':'france', 'sri':'sri lanka', 'gard':'france',
                        'mouilleron':'france', 'québec':'canada', 'atlantique nord-est':'norvège', 'fao 67':'états-unis',
                        'fao 61':'japon', 'soultzmatt':'france', 'cère':'france', 'mézières':'france', 'landes':'france',
                        'atlantique centre-est':'sénégal', 'bayonne':'france', 'revel':'france', 'fao 71':'indonésie',
                        'zeland':'nouvelle-zélande', 'figeac':'france', 'laval':'france', 'aquitaine':'france',
                        'tschechien':'république tchèque', 'pyrénées':'france', 'cotentin':'france', 'anjou':'france',
                        'auggen':'allemagne', 'amérique':'mexique', 'orme':'france', 'fa0 27':'norvège', 'suede':'suède',
                        'scotland':'royaume-uni', 'région centre':'france', 'trente':'italie', 'forez':'france',
                        'malaysie':'malaisie', 'sicile':'italie', 'benoît':'france', 'pacifique centre est':'états-unis',
                        'maromme':'france', 'morbihan':'france', 'pacifique nord-ouest':'japon', 'brasil':'brésil',
                        'ouest pacifique':'nouvelle-zélande', 'tcheque':'république tchèque', 'peru':'pérou',
                        'jamaika':'jamaïque', 'toscane':'italie', 'austria':'autriche', 'adeline':'france', 
                        'guérande':'france', 'mayenne':'france', 'kingdom':'royaume-uni', 'fao 51':'madagascar',
                        'noirmoutier':'france', 'est-centre':'france', 'giovanni':'italie', 'gers':'france',
                        'alba la romaine':'france', 'beuste':'france', 'vosge':'france', 'arcachon':'france',
                        'ventoux':'france', 'chateau':'france', 'vitell':'france', 'hépar':'france', 'montargis':'france',
                        'vermont':'france', 'belle':'france', 'việt nam':'vietnam', 'neuseeland':'nouvelle-zélande',
                        'uae':'émirats arabes unis', 'saverne':'france', 'ariège':'france', 'netherland':'pays-bas',
                        'ardèche':'france', 'jean':'france', 'marcel':'france', 'hawaï':'états-unis', 'abbaye':'france',
                        'salvador':'guatemala', 'norv':'norvège', 'itália':'italie', 'belique':'belgique',
                        'perou':'pérou', 'guyane':'guyane française', 'korea':'corée', 'beauregard':'france',
                        'fran':'france', 'jap':'japon', 'picardie':'france', 'igp':'france', 'laiterie':'france',
                        'betteville':'france', 'jura':'france', 'avelin':'france', 'armagnac':'france', "pays d'oc":'france',
                        'mont-dore':'france', 'nederland':'pays-bas', 'bordelais':'france', 'catalan':'espagne',
                        'coruna':'espagne', 'evian':'france', 'évian':'france', 'boulogne':'france', 'vallée des gaves':'france',
                        'angers':'france', 'norway':'norvège', 'chile':'chili', 'couëron':'bretagne', 'montclar':'france',
                        'crète':'grèce', 'bali':'indonésie', 'poska':'pologne', 'charente':'france', 
                        'macédoine':'macédoine du nord', 'ain':'france', 'languedoc':'france', 'vendée':'france',
                        'flandres':'belgique', 'saumur':'france', 'conserverie':'france', 'laiterie':'france', 
                        'écrins':'france', 'sweden':'suède', 'mexi':'mexique', 'cavaillon':'france', 'limousin':'france',
                        'genève':'suisse', '64290':'france', 'fromagerie':'france', 'gênes':'italie', 'alemanha':'allemagne',
                        'méxique':'mexique', 'compiègne':'france', "sud de l'europe":'italie', 'pays bas':'pays-bas',
                        'grenoble':'france', 'camargue':'france', 'zealand':'nouvelle-zélande', 'guilliers':'france', 
                        'orléan':'france', 'clairvic':'france', 'isr':'israël', 'tibet':'chine', 'plancoët':'france',
                        'argentina':'argentine', 'soultzmatt':'france', 'puys':'france', 'marceles':'france',
                        'gouzon':'france', 'rietberg':'allemagne', 'nillère':'france', 'villers':'france', 
                        'maulévrier':'france', 'vaucluse':'france', 'românia':'roumanie', 'atlantique centre-ouest':'mexique',
                        'bresse':'france', 'mont blanc':'france', 'tourouzelle':'france', 'source':'france', 'wissous':'france',
                        'léman':'suisse', 'carcassonne':'france', 'chelles':'france', 'grece':'grèce', 'greece':'grèce',
                        'malville':'france', 'himalaya':'chine', 'léognan':'france', 'île de ré':'france', 'bergues':'france',
                        'swaziland':'afrique du sud', 'montélimar':'france', 'améric':'mexique', 'kénya':'kenya', 
                        'tailandia':'thaïlande', 'saint ouen':'france', 'sarthe':'france', 'yunnan':'chine', 
                        'charenton':'france', 'méditerr':'grèce', 'cee':'roumanie', 'alsace':'france', 'alemania':'allemagne',
                        'marseille':'france', 'states':'états-unis', 'californie':'états-unis', 'montreuil':'france',
                        'gemenos':'france', 'danmark':'danemark', 'vill':'france', 'lithuanie':'lituanie', 
                        'griechenland':'grèce', 'denmark':'danemark', 'amsterdam':'pays-bas', 'gouvieux':'france',
                        'bourgogne':'france', 'bourgb':'france', 'bourge':'france', 'bocage':'france', 'abbé':'france',
                        'annecy':'france', 'flavigny':'france', 'sarbazan':'france', 'mesnay':'france', 'serbia':'serbie',
                        'fleurance':'france', 'aubagne':'france', 'louâtre':'france', 'dijon':'france', 'finland':'finlande',
                        'larressore':'france', 'delvert':'france', 'fécamp':'france', 'vertou':'france', 'limoges':'france',
                        'massegros':'france', 'россия':'russie'
                       }

dict_pays_df_cont = {'taïwan':'taïwan', 'royaume-uni':'royaume-uni','tchéquie':'république tchèque', 'viet':'vietnam', 
                     "états-unis":"états-unis"}

len(dict_pays_regroup_df), len(dict_pays_df_cont)


# In[95]:


# cellule utilisée pour analyser les différentes localisation contenant un certain mot-clé
clé = 'россия'
df['manufacturing_places'][df['manufacturing_places'].str.contains(clé).fillna(False)].value_counts()


# In[96]:


# cellule utilisée pour naviguer entre les différents pays pour vérifier leur orthographe, ou leur simple présence
# (pour remplacer un pays absent par un pays voisin par exemple : tibet, swaziland, etc.)
L_pays


# In[97]:


# fonction pour modifier les localisations selon le dictionnaire passé en argument
def replace_countries(dict_country, value):
    country = ''
    for key in dict_country.keys():
        contains = str(value).__contains__(key)
        if contains:
            country=dict_country[key]
            break
        else:
            country=value
    return country


# In[98]:


df_continents['Pays'] = df_continents['Pays'].apply(lambda x: replace_countries(dict_pays_df_cont, x))


# In[99]:


df['origins'] = df['origins'].apply(lambda x: replace_countries(dict_pays_regroup_df, x))


# In[100]:


df['manufacturing_places'] = df['manufacturing_places'].apply(lambda x: replace_countries(dict_pays_regroup_df, x))


# In[101]:


func = lambda x: False if x in df_continents['Pays'].tolist() else True
df['origins'][df['origins'].apply(func)].nunique(), df['origins'][df['origins'].apply(func)].count()/df['origins'].notna().sum()


# In[102]:


func = lambda x: False if x in df_continents['Pays'].tolist() else True
df['manufacturing_places'][df['manufacturing_places'].apply(func)].nunique(), df['manufacturing_places'][df['manufacturing_places'].apply(func)].count()/df['manufacturing_places'].notna().sum()


# A l'issu de notre travail de tri sur les localisations, nous sommes passés respectivement de <b>1828 à 228 (-87%) et de 1741 à 343 (-80%)</b> valeurs uniques dans les colonnes 'origins' et 'manufacturing_places', et les valeurs non manquantes de ces colonnes ne figurant pas dans la colonne 'Pays' de df_continents représenent respectivement 3,2% et ,1.4%.
# 
# <u>Nous trouvons ces taux acceptablement faibles pour les besoins de notre application</u>.
# 
# Nous définissons une dernière catégorie pour les pays en voisinage direct de la France:

# In[103]:


L_frontière = ['espagne', 'italie', 'belgique', 'luxembourg', 'allemagne', 'suisse', 'andorre', 'royaume-uni']


# In[104]:


# nous souhaitons associé à chaque localisation de notre jeu de donnée un continent, donc nous créons un dictionnaire assoc
dict_pays_cont = {}

for continent in df_continents['Continent'].value_counts().index:
    dict_pays_cont[continent] = df_continents[df_continents['Continent']==continent]['Pays'].str.strip().tolist()

for pays in L_frontière:
    dict_pays_cont['europe'].remove(pays)
    
dict_pays_cont['voisin']=L_frontière


# In[105]:


# fonction pour passer des pays aux continents
def country_to_continent(value):
    cont = ''
    if type(value) == float:
        if np.isnan(value):
            cont = np.nan
    else:
        for key in dict_pays_cont.keys():
            contains = value in dict_pays_cont[key]
            if contains:
                cont = key
                break
            else:
                cont = 'inconnue'
    return cont


# In[106]:


df['origins'] = df['origins'].apply(country_to_continent)
df['manufacturing_places'] = df['manufacturing_places'].apply(country_to_continent)


# Nous n'avons désormais plus besoin de notre dataframe <b>df_continents</b>.

# In[107]:


del df_continents


# <b>Intéressons nous maintenant à la colonne 'labels_fr'</b>

# In[108]:


df['labels_fr'] = df['labels_fr'].str.lower()


# In[109]:


df['labels_fr'].value_counts()[:10]


# In[110]:


# cellule utilisée pour analyser les différentes valeurs de labels contenant certains mots-clés
clé1 = 'personne'
clé2 = ''
df['labels_fr'].loc[df['labels_fr'].str.contains(clé1).fillna(False)].loc[df['labels_fr'].str.contains(clé2).fillna(False)].value_counts()


# Après un peu de recherche sur les différents labels, et les organismes délivrant ces labels (et le sérieux de ces organismes), nous ne retiendrons que les labels suivants : 
# 
# - Bio européen (ecocert, fr-bio, certipaq, certisud, agriculture biologique)
# - Label qualité (aop, igp, stg, aoc, label rouge)
# - Gestion durable (ressources : fsc, rainforest alliance, utz, msc ; déchêts : point vert, eco emballage)
# - Equitable (fairtrade international, max havelaar, fsc)
# 
# Nous allons également catégoriser les produits selon les critères suivants :
# 
# - Conformité Halal
# - Conformité Kascher
# - Présence d'OGM
# - Convient aux végétariens
# - Convient aux végétaliens
# - Teneur réduite en sel
# - Teneur réduite en sucres
# - Déconseillé aux femmes enceintes
# - Déconseillé à certaines catégories de personnes
# 
# Nous allons donc créer un dictionnaire composé de 14 associations clé:valeurs pour ensuite créer un tableau disjonctif complet sur les catégories citées ci-avant.

# In[111]:


set_bio_euro = set(['ecocert', 'fr-bio', "agriculture biologique", "agriculture-biologique"])

set_label_quali = set(['aop', 'igp', 'stg', 'aoc', "label rouge", "label-rouge"])

set_gestion_dur = set(['fsc', 'rainforest', 'utz', 'msc', "point vert", "point-vert", "eco emballage", "eco-emballage"])

set_equit = set(['fairtrade', 'havelaar'])

set_halal = set(['halal'])

set_kascher = set(['kascher', 'kosher', 'cacher'])

set_ogm = set(['sans ogm', 'sans-ogm', 'non-ogm'])

set_vegetariens = set(['végétarien', 'vegetar'])

set_vegan = set(['vegan', 'végétalien'])

set_sel_reduit = set(['sel'])

set_sucres_reduit = set(['sucre'])

set_femmes_enceintes = set(['enceinte'])

set_personnes = set(['personne'])

dict_labels = {'bio_europe':set_bio_euro,'label_qualité':set_label_quali, 'gestion_durable':set_gestion_dur, 
               'commerce_équitable':set_equit, 'halal':set_halal, 'kascher':set_kascher, 'ogm':set_ogm, 
               'végétariens':set_vegetariens, 'végétaliens':set_vegan, 'sel_réduit':set_sel_reduit, 
               'sucres_réduits':set_sucres_reduit, 'femmes_enceintes':set_femmes_enceintes, 
               'catégories_personnes':set_personnes}


# In[112]:


# nous définissons la fonction qui nous indiquera si un produit apaprtient à une des catégories définies ci-avant
def belong_to_label(cat_label, value):
    dict_cat = dict_labels[cat_label]
    if type(value) == float:
        if np.isnan(value):
            n = np.nan
    else:
        for label in dict_cat:
            contains = str(value).__contains__(label)
            if contains:
                n = 1
                break
            else:
                n = 0
    return n

# fonction permettant de créer les colonnes de labels
def set_cols_labels():
    k=1
    func = lambda value: belong_to_label(key, value)
    for key in dict_labels.keys():
        if key in df.columns.tolist():
            df.drop(key, axis=1, inplace=True)
        df.insert(loc=(df.columns.get_loc('labels_fr')+k), column=key, value=df['labels'].apply(func)) 
        k+=1


# In[113]:


set_cols_labels()


# In[114]:


df.loc[:,'bio_europe':'catégories_personnes']


# In[115]:


df_2 = df.copy()


# ### RAF partie 2 :
# - vérifier les données aberrantes des colonnes

# #### 3. Partie 'constitution du produit'

# In[116]:


df = df_2.copy()


# In[117]:


col3 = list(df.columns[df.columns.get_loc('ingredients_text'):df.columns.get_loc('serving_size')])
col3


# In[118]:


df[col3]


# In[119]:


df[col3].isna().mean()


# In[120]:


df[col3].nunique()


# A ce stade nous savons que nous allons supprimer la colonne <u>'allergens_fr'</u>, <u>'traces_tags'</u> qui nous fournit une information similaire à celle de <u>'traces_fr'</u> mais en anglais et dans une mise en forme moins lisible et <u>'traces'</u> qui nous fournit une information moins synthétique que <u>'traces_fr'</u> (plus de valeurs uniques pour le même taux de remplissage).
# 
# Notons que les colonnes <u>'allergens'</u> et <u>'traces_fr'</u> contiennent le même type d'information : la présence de substances pouvant provoquer une allergie.
# 
# - Nous combinerons donc ces 2 colonnes en espérant ainsi obtenir une information plus complète.
# 
# Mais avant cela, concernant la colonne <u>'ingredients_text'</u>, nous allons compter le nombre d'éléments dans la liste d'ingrédients de chaque produit dans une nouvelle colonne <u>'ingredients_n'</u> et conserver la colonne pour pouvoir la montrer aux utilisateurs finaux de notre applications.

# In[121]:


df.drop(['allergens_fr','traces_tags','traces'], axis=1, inplace=True)


# In[122]:


nb_ingredients = lambda x: len(str(x).split(',')) if (str(x)!='nan') else np.nan
df.insert(loc=(df.columns.get_loc('ingredients_text')+1), column='ingredients_n', value=df['ingredients_text'].apply(nb_ingredients))


# In[123]:


df[['ingredients_text','ingredients_n']]


# In[124]:


col3 = list(df.columns[df.columns.get_loc('ingredients_text'):df.columns.get_loc('serving_size')])
df[col3].isna().mean()


# Nous commencçons notre travail sur les colonnes <b>traces_fr</b> et <b>allergens</b> :

# In[125]:


# pour faciliter le travail sur les chaînes de caractères, on les passe toutes en minuscule
df['traces_fr'] = df['traces_fr'].str.lower()
df['allergens'] = df['allergens'].str.lower()


# In[126]:


L_traces = get_list_splits_str('traces_fr')
len(L_traces)


# In[127]:


L_traces


# In[128]:


L_allergens = get_list_splits_str('allergens')
len(L_allergens)


# In[129]:


L_valeurs_allerg = df['traces_fr'].value_counts().index.tolist()+df['allergens'].value_counts().index.tolist()
ens_valeurs_allerg = set(L_valeurs_allerg)
len(ens_valeurs_allerg)


# In[130]:


L_traces_uniques = get_list_uniques_splits_str('traces_fr')
len(L_traces_uniques)


# In[131]:


L_traces_uniques


# In[132]:


L_traces_reduit = ['blé', 'wheat','gluten', 'orge','cereales', 'epautre', 'cereals', 'glurent', 
                   
                   'céléri','celeria', 'czeleri', 'selleri', 'celerie', 
                   
                   'oeuf', 'œufs', 'egg', 'eggs',
                   
                   'coque', 'pistache', 'amande', 'noisette', 'noix','guscio', 'nusse', 'pistachio', 'haselnuss', 'secos', 
                   'nut', 'cashewnusse', 'pekannusse', 'amendoa', 'casca',
                   
                   'lupin', 'lupino',
                   
                   'lait','lactiques','lactosérum', 'milk', 'lactose', 'creme', 'beurre', 'laitier', 'laiit','lactoserum',
                   
                   'moutarde', 'mustard', 'mouarde', 'moutrde', 
                   
                   'poisson', 'fish','sardines','thon', 'crevettes','fisch', 'pesce', 
                   
                   'crustacés,', 'crustaces', 'surimi', 'crabe', 'crustacei', 'curstaces', 'crustacee', 
                   'crustacees', 'drustace',
                   
                   'mollusques', 'molluschi', 'jacques',
                   
                   'disulfite','sulfites', 'sulfates', 'sulfureux',
                   
                   'arachide', 'arachides', 'cacahuètes', 'cacahetes',
                   
                   'sesame', 'sésame', 'susam','cesame', 'sesamo',
                   
                   'soja', 'sija'
                    ]
len(L_traces_reduit)


# In[133]:


L_allerg_uniques = get_list_uniques_splits_str('allergens')
len(L_allerg_uniques)


# In[134]:


L_allerg_uniques


# In[135]:


L_allerg_reduit = ['comté', 'milch', 'vollmilchpulver', 'butterreinfett', 'magermilchpulver', 'fromage', 'emmental', 'lctosa',
                   'laitière', 'roquefort', 'creme', 'pecorino', 'parmigiano', 'milchzucker', 'milcheiweißhydrolysat', 'gouda',
                   'edam', 'actosérum', 'mozzarella', 'raclette', 'ricotta', 'tome', 'cheddar', 'milchschokolade', 'crème',
                   'molkenpulver', 'parmesan', 'maroilles', 'sahnepulver', 'butter', 'magermilchjoghurtpulver', 
                   'vollmilchpulver', 'leite', 'iactose', 'beaufort','yaourt', 'magermilch', 'yaourts' ,'sahnepulver', 'édam',
                   'présure', 'mascarpone', 'latte', 'feta', 'mimolette', 'laktose', 'iait', 'fromage', 'milchserum',
                   'reblochon', 'eiweißpulver', 'magermilchjoghurtpulver', 'milcheiweiß', 'magermilchkonzentrat',
                   'milchserumkonzentrat', 'whey', 'cantal', 'leche', 'lactosa', 'mantequilla', 'kuhmilch', 'weichkäse',
                   'süßmlkenpulver', 'molke', 'magermllchpulver', 'gorgonzola', 'crème', 'laitiers', 'iactosèrum', 
                   'emmenthal', 'cream', 'lactate', 'beure', 'magemilchpulver', 'bleu', 'kondensmagermilch', 'caséinate', 
                   'gouda', 'écrémé', 'schlagsahne', 'milcheiweißpulver', 'vollfett-frischkäse', 'schlagsahne', 'reblochon',
                   'ziegenmilch', 'tomme', 'laitiére', 'laitiéres', 'ialt', 'lactique', 'iactoserum', 'pecorino', 'ferments',
                   'ferment', 'lacto', 'caséinates', 'maroilles', 
                   
                   'avoine','seigle', 'épeautre', 'son', 'glutn', 'barley','gerstenmalzextrakt', 'froment', 'weizenmehl',
                   'weizenstärke', 'weizen-reis-extrudat', 'weizen-reis-extrudat', 'weizenvollkornmehl', 'segale', 'orzo',
                   'avena', 'weizeneiweiß', 'frumento', 'vollkornhaferflocken', 'volkornweizenflocken',
                   'vollkorngerstenflocken', 'cebada', 'gerste', 'weichweizenmehl', 'hartweizengrieß', 
                   'roggenmehl', 'trigo', 'gerstenflocken', 'weizen', 'gerstenmalz', 'weizenflocken', 'couscous', 
                   'gerstenmalzmehl', 'weizenmalzmehl', 'blés', 'boulghour', 'gerstenvollkornmehl', 'gerstenvollkornmehl',
                   'hafervollkornmehl', 'dinkelvollkornmehl', 'roggenvollkornmehl', 'hafervollkornflocken', 
                   'weizenvollkornflocken', 'weizenkleber', 'hartweizengrieß', 'millet', 'siegle', 'peanuts', 'malté', 
                   'weizengluten', 'amidon', 'glúten', 
                   
                   'cajou', 'pécan','pin','pignon', 'mandeln', 'pecan', 'haselnüsse', 'amandons', 'haselnussmasse', 'haselnüsse',
                   'noisettes', 'haselnuskern', 'haselnussmark', 'mandeln', 'cashewkerne', 'almendras', 'almonds', 'hazelnut',
                   'avelãs', 'cashews', 'amendes', 'nuts'
                   
                   
                   'cabillaud', 'saumon', 'maquereau', 'colin', 'brochet', 'écrevisses', 'limande','poissons', 'truite', 'lieu',
                   'anchois', 'homard', 'maquereaux', 'langoustines', 'morue', 'esturgeon', 'mer', 'merlu', 'gambas', 'merlan',
                   'bar', 'rouget', 'barbet', 'langoustine', 'harengs', 'hareng', 'sardine', 
                   
                   'huitre', 'clams', 'coquillages', 'huître', 'tourteau', 'crustace', 'moule', 'bulots', 
                   
                   'pulpe', 'poulpe', 'encornet', 'calamars', 'calmars', 'seiche', 'encornets', 'seiches', 
                   
                   'soybeans', 'lécithine', 'lecithin', 'sojalecithin', 'sojakerne', 'sojalecithine', 'soia', 
                   'lécithine de soja', 'sojasoßenpulver', 'sojabohnen', 'tofu', 'mungo', 'soya', 'soy',
                   
                   'hühnerei-trockeneiweiß','uovo', 'hühnervolleipulver', 'hühnerei', 'eigelb', 'huevo',
                   
                   'sesamöl',
                   
                   'senf', 'moutard', 
                   
                   'schwefeldioxid', 'sulfates','sulfito',
                   
                   'cacahouètes', 'erdnüsse', 'cacahuète', 
                   
                   'céleris', 'țelină'
                  ]
len(L_allerg_reduit)


# In[136]:


ens_allerg_uniques = set(L_traces_uniques)
for element in L_allerg_uniques:
    ens_allerg_uniques.add(element)
len(ens_allerg_uniques)


# In[137]:


ens_allerg_reduit = set(L_traces_reduit)
for element in L_allerg_reduit:
    ens_allerg_reduit.add(element)
len(ens_allerg_reduit)


# Nous allons réaliser un tableau disjonctif complet où chaque produit appartiendra à autant de catégories d'allergènes qu'il en contient. Nous ne pouvons donc pas utiliser la fonction OneHotEncoder de scikit-learn (qui ne peut attribuer qu'une modalité à chaque individu).
# Nous allons créer un ensemble de susbstances allergènes par catégorie (14 en tout), et créer une colonne pour chacune des catégories, et nous vérifierons pour chaque produit la présence de substances de la catégorie dans son champ 'allergens' et 'traces_fr'.
# 
# Le travail préliminaire réalisé ci-avant a permis de passer de 11788 valeurs différentes, à 1879 modalités uniques puis à 309 modalités discriminant 14 catégories de substances allergènes (division par 38).
# 
# NB : le tri des "mots-clés" des listes de modalités uniques pour obtenir les listes réduites a été réalisé à la main, et bien que l'opération ait été réalisée minutieusement, il y aura des manques que nous considérons comme étant des erreurs de perte d'information inhérentes au processus de transformation que nous avons choisi.
# - Une autre manière de procéder aurait été de lister l'ensemble des substances les plus courantes appartenant aux 14 catégories d'allergènes qu'il est obligatoire de mentionner sur l'emballage d'un produit, et de les traduire dans toutes les langues de l'union européenne, mais cela aurait donné un tableau trop volumineux et nous n'aurions pas pu capter les mots-clés relevant de fautes d'orthographes, les variations de mots avec ou sans accent, ainsi que les mots au pluriel comme nous avons pu le faire ici.

# In[138]:


set_gluten = set(['gluten', 'glurent', 'glutn', 'glúten'
    
])

set_oeuf = set(['hühnerei-trockeneiweiß','uovo', 'hühnervolleipulver', 'hühnerei', 'eigelb', 'huevo', 'oeuf', 'œufs', 'egg',
                'eggs'])

set_fruits_coque = set(['coque', 'pistache', 'amande', 'noisette', 'noix','guscio', 'nusse', 'pistachio', 'haselnuss', 'secos', 
                   'nut', 'cashewnusse', 'pekannusse', 'amendoa', 'casca', 'cajou', 'pécan','pin','pignon', 'mandeln', 'pecan', 
                   'haselnüsse', 'amandons', 'haselnussmasse', 'haselnüsse','noisettes', 'haselnuskern', 'haselnussmark',
                   'mandeln', 'cashewkerne', 'almendras', 'almonds', 'hazelnut', 'avelãs', 'cashews', 'amendes', 'nuts'])

set_lupin = set(['lupin', 'lupino'])

set_lait = set(['lait','lactiques','lactosérum', 'milk', 'lactose', 'creme', 'beurre', 'laitier', 'laiit','lactoserum',
                'comté', 'milch', 'vollmilchpulver', 'butterreinfett', 'magermilchpulver', 'fromage', 'emmental', 'lctosa',
                'laitière', 'roquefort', 'creme', 'pecorino', 'parmigiano', 'milchzucker', 'milcheiweißhydrolysat', 'gouda',
                'edam', 'actosérum', 'mozzarella', 'raclette', 'ricotta', 'tome', 'cheddar', 'milchschokolade', 'crème',
                'molkenpulver', 'parmesan', 'maroilles', 'sahnepulver', 'butter', 'magermilchjoghurtpulver', 
                'vollmilchpulver', 'leite', 'iactose', 'beaufort','yaourt', 'magermilch', 'yaourts' ,'sahnepulver', 'édam',
                'présure', 'mascarpone', 'latte', 'feta', 'mimolette', 'laktose', 'iait', 'fromage', 'milchserum',
                'reblochon', 'eiweißpulver', 'magermilchjoghurtpulver', 'milcheiweiß', 'magermilchkonzentrat',
                'milchserumkonzentrat', 'whey', 'cantal', 'leche', 'lactosa', 'mantequilla', 'kuhmilch', 'weichkäse',
                'süßmlkenpulver', 'molke', 'magermllchpulver', 'gorgonzola', 'crème', 'laitiers', 'iactosèrum', 
                'emmenthal', 'cream', 'lactate', 'beure', 'magemilchpulver', 'bleu', 'kondensmagermilch', 'caséinate', 
                'gouda', 'écrémé', 'schlagsahne', 'milcheiweißpulver', 'vollfett-frischkäse', 'schlagsahne', 'reblochon',
                'ziegenmilch', 'tomme', 'laitiére', 'laitiéres', 'ialt', 'lactique', 'iactoserum', 'pecorino', 'ferments',
                'ferment', 'lacto', 'caséinates', 'maroilles'])

set_sulfites = set(['schwefeldioxid', 'sulfates','sulfito', 'disulfite','sulfites', 'sulfates', 'sulfureux'])

set_poissons = set(['poisson', 'fish','sardines','thon', 'crevettes','fisch', 'pesce','cabillaud', 'saumon', 'maquereau', 
                    'colin', 'brochet', 'écrevisses', 'limande','poissons', 'truite', 'lieu', 'anchois', 'homard',
                    'maquereaux', 'langoustines', 'morue', 'esturgeon', 'mer', 'merlu', 'gambas', 'merlan', 'bar', 'rouget',
                    'barbet', 'langoustine', 'harengs', 'hareng', 'sardine'])

set_mollusques = set(['mollusques', 'molluschi', 'jacques','pulpe', 'poulpe', 'encornet', 'calamars', 'calmars', 'seiche', 
                      'encornets', 'seiches'])

set_crustaces = set(['crustacés,', 'crustaces', 'surimi', 'crabe', 'crustacei', 'curstaces', 'crustacee', 
                     'crustacees', 'drustace','huitre', 'clams', 'coquillages', 'huître', 'tourteau', 'crustace',
                     'moule', 'bulots'])

set_soja = set(['soybeans', 'lécithine', 'lecithin', 'sojalecithin', 'sojakerne', 'sojalecithine', 'soia', 'lécithine de soja',
                'sojasoßenpulver', 'sojabohnen', 'tofu', 'mungo', 'soya', 'soy','soja', 'sija'])

set_cereales = set(['blé', 'wheat','gluten', 'orge','cereales', 'epautre', 'cereals', 'glurent', 'avoine','seigle', 
                    'épeautre', 'son', 'glutn', 'barley','gerstenmalzextrakt', 'froment', 'weizenmehl', 'weizenstärke',
                    'weizen-reis-extrudat', 'weizen-reis-extrudat', 'weizenvollkornmehl', 'segale', 'orzo', 'avena', 
                    'weizeneiweiß', 'frumento', 'vollkornhaferflocken', 'volkornweizenflocken', 'vollkorngerstenflocken',
                    'cebada', 'gerste', 'weichweizenmehl', 'hartweizengrieß', 'roggenmehl', 'trigo', 'gerstenflocken',
                    'weizen', 'gerstenmalz', 'weizenflocken', 'couscous', 'gerstenmalzmehl', 'weizenmalzmehl', 'blés',
                    'boulghour', 'gerstenvollkornmehl', 'gerstenvollkornmehl', 'hafervollkornmehl', 'dinkelvollkornmehl',
                    'roggenvollkornmehl', 'hafervollkornflocken', 'weizenvollkornflocken', 'weizenkleber', 'hartweizengrieß',
                    'millet', 'siegle', 'malté', 'weizengluten', 'amidon', 'glúten'])

set_arachides = set(['arachide', 'arachides', 'cacahuètes', 'cacahetes','cacahouètes', 'erdnüsse', 'cacahuète', 'peanuts'])

set_celeri = set(['céléri','celeria', 'czeleri', 'selleri', 'celerie', 'céleris', 'țelină'])

set_sesame = set(['sesame', 'sésame', 'susam','cesame', 'sesamo', 'sesamöl'])

set_moutarde = set(['moutarde', 'mustard', 'mouarde', 'moutrde','senf', 'moutard'])

dict_allerg = {'gluten': set_gluten, 'oeuf':set_oeuf, 'fruits_coque':set_fruits_coque, 'lupin':set_lupin, 'lait':set_lait, 
               'sulfites':set_sulfites, 'poissons':set_poissons, 'mollusques':set_mollusques, 'crustaces':set_crustaces, 
               'soja':set_soja, 'cereales':set_cereales, 'arachides':set_arachides, 'celeri':set_celeri, 'sesame':set_sesame, 
               'moutarde':set_moutarde}


# In[139]:


for key in dict_allerg.keys():
    print(key)


# In[140]:


len(dict_allerg.keys())


# In[141]:


for substance in dict_allerg['oeuf']:
    print(substance)


# In[142]:


df.insert(loc=(df.columns.get_loc('traces_fr')+1), column='substances_allergenes', value=(df['traces_fr'] + ',' + df['allergens']))


# In[143]:


df['substances_allergenes']


# In[144]:


# nous définissons la fonction qui nous indiquera si un produit contient une substance de la catégorie passée en argument
def contains_allerg(cat_allerg, value):
    dict_cat = dict_allerg[cat_allerg]
    if type(value) == float:
        if np.isnan(value):
            n = np.nan
    else:
        for substance in dict_cat:
            contains = str(value).__contains__(substance)
            if contains:
                n = 1
                break
            else:
                n = 0
    return n

# fonction permettant de créer les colonnes de catégories d'allergènes
def set_cols_cat_allerg():
    k=1
    func = lambda value: contains_allerg(key, value)
    for key in dict_allerg.keys():
        if key in df.columns.tolist():
            df.drop(key, axis=1, inplace=True)
        df.insert(loc=(df.columns.get_loc('traces_fr')+k), column=key, value=df['substances_allergenes'].apply(func)) 
        k+=1


# In[145]:


set_cols_cat_allerg()


# In[146]:


df[df['substances_allergenes'] != 'inconnues'].loc[:,'oeuf':'moutarde']


# In[147]:


df.shape


# In[148]:


(df[df['substances_allergenes'].notna()].loc[:,'oeuf':'moutarde'].sum(axis=1) == 0).value_counts()


# Nous avons perdu l'information de présence d'allergènes pour uniquement 7 produits sur la totalité (0.06%).
# Nous traitons ces produits réticents à la main.

# In[149]:


df[df['substances_allergenes'].notna()][df[df['substances_allergenes'].notna()].loc[:,'oeuf':'moutarde'].sum(axis=1) == 0]['substances_allergenes']


# Nous ajoutons la valeur 'céleri' à notre ensemble 'set_celeri'.

# In[150]:


set_celeri.clear()
set_celeri = set_celeri = set(['céléri','celeria', 'czeleri', 'selleri', 'celerie', 'céleris', 'țelină', 'céleri'])
dict_allerg = {'oeuf':set_oeuf, 'fruits_coque':set_fruits_coque, 'lupin':set_lupin, 'lait':set_lait, 'sulfites':set_sulfites,
               'poissons':set_poissons, 'mollusques':set_mollusques, 'crustaces':set_crustaces, 'soja':set_soja,
               'cereales':set_cereales, 'arachides':set_arachides, 'celeri':set_celeri, 'sesame':set_sesame, 
               'moutarde':set_moutarde}


# In[151]:


df['celeri'] = df['substances_allergenes'].apply(lambda value: contains_allerg('celeri', value))


# In[152]:


(df[df['substances_allergenes'] != 'inconnues'].loc[:,'oeuf':'moutarde'].sum(axis=1) == 0).value_counts()


# On retrouve notre ligne mentionnant du 'réglisse' qui n'est pas une substance allergène à mentionner obligatoirement par le fabricant. Nous n'avons désormais plus besoin des colonnes 'traces_fr' et 'allergens'.

# In[153]:


df.drop(['traces_fr','allergens'], axis=1, inplace=True)


# In[154]:


df.shape


# In[155]:


df_3 = df.copy()


# #### 4. Partie 'informations diverses'

# In[156]:


df = df_3.copy()


# In[157]:


col4 = list(df.columns[df.columns.get_loc('serving_size'):df.columns.get_loc('energy_100g')])
col4


# In[158]:


df[col4].isna().mean()


# In[159]:


#cellule utilisée pour explorer les différentes valeurs des colonnes
df['additives_fr'].value_counts()[:15]


# Nous n'allons conserver que les colonnes suivantes :
# - 'additives_n' : qui nous renseigne sur le nombre d'additifs présents dans le produit
# - 'additives_fr' : qui nous renseignent sur les additifs présents (pour évaluer leur dangerosité)
# - 'ingredients_from_palm_oil_n' : qui nous renseigne sur le nombre d'ingrédients issus de l'huile de palme, présents dans le produit
# - 'ingredients_that_may_be_from_palm_oil_n' : qui nous renseigne sur le nombre d'ingrédients peut-être issus de l'huile de palme, présents dans le produit
# - 'nutrition_grade_fr' : qui nous renseigne sur le NUTRISCORE du produit
# - 'image_small_url' : qui nous fournit l'url de l'image du produit que l'on pourra afficher dans notre application
# 
# Les autres colonnes sont soit vides, soit nous donnent une informations que nous n'allons pas exploiter, ou bien sont moins compréhensible ou synthétique.

# In[160]:


col_part4_to_drop = col4
col_part4_to_drop.remove('additives_n')
col_part4_to_drop.remove('additives_fr')
col_part4_to_drop.remove('ingredients_from_palm_oil_n')
col_part4_to_drop.remove('ingredients_that_may_be_from_palm_oil_n')
col_part4_to_drop.remove('nutrition_grade_fr')
col_part4_to_drop.remove('image_small_url')
df.drop(columns=col_part4_to_drop, inplace=True)


# In[161]:


col4 = list(df.columns[df.columns.get_loc('additives_n'):df.columns.get_loc('energy_100g')])
df[col4].head()


# In[162]:


df[col4].isna().mean()


# <b>Travail sur les additifs</b>
# 
# Nous importons un jeu de données informatif de la dangerosité des additifs alimentaires : ici ne sont présents que les additifs avec un niveau de dangerosité pour la santé suffisant pour que la question du bien fondé de la consommation régulière du produit se pose. Tous les additifs absents de cette liste auront un niveau de dangerosité égal à 0, tandis que les additifs de ce jeu de donnée auront un niveau de dangerosité allant de 1 à 3.

# In[163]:


df_additifs = pd.read_csv("extradatas\\additifs_dangereux.csv", delimiter=';', on_bad_lines='skip')
df_additifs


# A ce stade nous voulons récupérer la liste des codes des additifs contenus dans chaque produit pour facilement faire la correspondance avec notre jeu de données sur la dangerosité des additifs.

# In[164]:


df['additives_fr'].value_counts()[10:]


# In[165]:


selector = lambda x: [str(element).split(' -')[0] for element in str(x).split(',')] if type(x) != float else np.nan if np.isnan(x) else x
df.insert(loc=(df.columns.get_loc('additives_fr')+1), column='codes_additifs', value=df['additives_fr'].apply(selector))


# In[166]:


df['codes_additifs']


# In[167]:


def highest_danger_level(value):
    i=0
    liste_code_danger_add = df_additifs['Id_additif'].tolist()
    list_level_danger_add = df_additifs['Niveau_danger'].tolist()
    if type(value) == float:
        if np.isnan(value):
            danger = np.nan
    else:
        for code_add in value:
            danger=0
            contains = liste_code_danger_add.__contains__(code_add)
            if contains:
                if list_level_danger_add[i] > danger :
                    danger = list_level_danger_add[i]
            i+=1
    return danger


# In[168]:


df.insert(loc=(df.columns.get_loc('codes_additifs')+1), column='additif_niveau_danger', value=df['codes_additifs'].apply(highest_danger_level))


# Nous sommes rassurés par la présence d'additifs de dangerosité du plus bas niveau uniquement.
# 
# Nous n'avons désormais plus besoin de notre dataframe <b>df_additifs</b>.

# In[169]:


del df_additifs


# In[170]:


df_4 = df.copy()


# #### 5. Partie informations nutritionnelles

# In[171]:


df = df_4.copy()


# In[172]:


col5 = list(df.columns[df.columns.get_loc('energy_100g'):])
col5


# In[173]:


len(col5)


# In[174]:


msno.bar(df[col5], sort='descending')


# Tout d'abord nous allons créer une colonne indiquant si un produit contient de l'alcool:

# In[175]:


df.insert(loc=(df.columns.get_loc('energy_100g')-1), column='alcool', value=df['alcohol_100g'].apply(lambda x: 1 if x>0.0 else 0))


# In[176]:


df['alcool'].value_counts()


# Pour la suite de notre étude, nous n'allons garder que les colonnes présentant un <b>taux de remplissage supérieur à 45%</b> et disposant d'un <b>nombre de valeurs uniques supérieur à 50</b>, afin de pouvoir réaliser une analyse statistique valable.

# In[177]:


col5 = np.intersect1d(df[col5].columns[df[col5].nunique()>50],df[col5].columns[df[col5].notna().sum()>45*df.shape[0]/100]).tolist()


# In[178]:


df[col5]


# In[179]:


df.drop(columns=np.setdiff1d(df.columns[df.columns.get_loc('energy_100g'):],col5), inplace=True)


# In[180]:


df.drop('nutrition-score-uk_100g', axis=1, inplace=True)
col5.remove('nutrition-score-uk_100g')


# In[181]:


col5


# In[182]:


col6 = col5.copy()
col6.remove('nutrition-score-fr_100g')
col6.remove('energy_100g')


# In[183]:


for col in col6:
    df[col].clip(0,100, inplace=True)


# In[184]:


df[col5].describe()


# In[185]:


df.shape


# In[186]:


df.isna().mean().mean()


# ## Traitement des valeurs manquantes

# In[ ]:




