__doc__ = """Ce module contient la définition des fonctions utiles dans le cadre de la réalisation d'une ACP :
	- display_scree_plot : qui affiche l'éblouis des valeurs propres avec la somme cumulée des inerties des composantes
	- display_factorial_planes : qui affiche la projection des individus dans le plan factoriel choisi
	- display_corr_circle : qui affiche le cercle des corrélations dans le plan factoriel choisi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def display_scree_plot(pca):

    """
    Affiche l'éblouis des valeurs propres de l'ACP passé en argument.
    """ 

    x_list = list(range(1,(len(pca.components_)+1)))
    Components_inertia = 100*pca.explained_variance_ratio_
    Cumsum_inertia = Components_inertia.cumsum()
    
    plt.figure(figsize=(8,6))
    
    bar = plt.bar(x_list,Components_inertia, )
    plt.bar_label(bar, 
                  labels=np.round(Components_inertia,0), 
                  padding=3, 
                  color='black', 
                  fontsize=12)
    plt.plot(x_list,Cumsum_inertia, marker='o', color='red')
    for x, y, text in zip(x_list, Cumsum_inertia, np.round(Cumsum_inertia,0)):
        if x !=1:
            plt.text(x, 
                     y*(0.9), 
                     text, 
                     ha = 'left',
                     bbox = dict(facecolor = 'red', alpha =0.3))

    plt.title("Éblouis des valeurs propres", size=18)
    plt.xlabel("Rang des composantes d'inertie", size=15)
    plt.ylabel("Proportion d'intertie expliquée", size=15);


def display_factorial_planes(   X_projected, 
                                x_y, 
                                pca=None, 
                                labels = None,
                                clusters=None,
				dict_colors=None,
				dict_order=None,
                                alpha=1,
                                figsize=[10,8], 
                                marker="."):
    """
    Affiche la projection des individus

    Positional arguments : 
    -------------------------------------
    X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

    Optional arguments : 
    -------------------------------------
    pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
    labels : list ou tuple : les labels des individus à projeter, default = None
    clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
    alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8] 
    marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
    dict_colors : dictionnaire associant chaque cluster à une couleur, default = None
    """

    # Transforme X_projected en un df
    X_ = pd.DataFrame(X_projected)
    
    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize: 
        figsize = (7,6)

    # On vérifie la variable axis 
    if not len(x_y) ==2 : 
        raise AttributeError("2 axes sont demandées")   
    if max(x_y )>= X_.shape[1] : 
        raise AttributeError("la variable axis n'est pas bonne")   

    # On définit x et y 
    x, y = x_y

    # Initialisation de la figure       
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # On rajoute la color, les clusters et les labels à X_
    X_["clusters"] =  clusters if clusters is not None else "None" 
    X_["labels"] =  labels if labels is not None else "None"
    c_unique_list = X_["clusters"].sort_values().unique()
    
    if dict_order is not None:
        ord_dict = dict_order
        X_["order"] = X_["clusters"].apply(lambda i : ord_dict[i])

    if dict_colors is None:
        c_dict = {j:i+1 for i, j in enumerate(c_unique_list)}
        X_["colors"] = X_["clusters"].apply(lambda i : c_dict[i])

        # Pour chaque couleur / cluster
        for c in sorted(X_.clusters.unique()) : 
            # On selectionne le sous DF
            sub_X =X_.loc[X_.clusters == c, : ]

            # Clusters and color
            cluster = sub_X.clusters.iloc[0]
            color = sub_X.colors.iloc[0]
            
            if dict_order is None:
                # On affiche les points
                ax.scatter( sub_X.iloc[:, x], 
                            sub_X.iloc[:, y], 
                            alpha=alpha, 
                            label = cluster ,
                            cmap="Set1", 
                            marker=marker)
            else:
                order = sub_X.order.iloc[0]
                
                # On affiche les points
                ax.scatter( sub_X.iloc[:, x], 
                            sub_X.iloc[:, y], 
                            alpha=alpha, 
                            label = cluster ,
                            cmap="Set1", 
                            marker=marker,
                            zorder=order)
        
    else:
        c_dict = dict_colors

        X_["colors"] = X_["clusters"].apply(lambda i : c_dict[i])

        # Pour chaque couleur / cluster
        for c in sorted(X_.clusters.unique()) : 
            # On selectionne le sous DF
            sub_X =X_.loc[X_.clusters == c, : ]

            # Clusters and color
            cluster = sub_X.clusters.iloc[0]
            color = sub_X.colors.iloc[0]

            if dict_order is None:
                # On affiche les points
                ax.scatter( sub_X.iloc[:, x], 
                            sub_X.iloc[:, y], 
                            alpha=alpha, 
                            label = cluster ,
                            c=color, 
                            marker=marker)
            else:
                order = sub_X.order.iloc[0]
                
                # On affiche les points
                ax.scatter( sub_X.iloc[:, x], 
                            sub_X.iloc[:, y], 
                            alpha=alpha, 
                            label = cluster ,
                            c=color, 
                            marker=marker,
                            zorder=order)

        
    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe 
    if pca : 
        v1 = str(round(100*pca.explained_variance_ratio_[x], 1))  + " %"
        v2 = str(round(100*pca.explained_variance_ratio_[y], 1))  + " %"
        v3 = str(round(100*(pca.explained_variance_ratio_[y]+pca.explained_variance_ratio_[x]), 1))  + " %"
    else : 
        v1=v2= ''

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f'F{x+1} ({v1})')
    ax.set_ylabel(f'F{y+1} ({v2})')

    # Valeur x max et y max
    x_max = np.abs(X_.iloc[:, x]).max() *1.1
    y_max = np.abs(X_.iloc[:, y]).max() *1.1

    # On borne x et y 
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom= -y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
    plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)

    # Affichage des labels des points
    if labels : 
        #j'ai copié collé la fonction sans la lire
        for i,(_x,_y) in enumerate(X_[:,[x,y]]):
            plt.text(_x, _y, labels[i], fontsize='14', ha='center',va='center') 
    
    # Titre, legend et display
    plt.title(f"Projection des individus sur F{x+1} et F{y+1} ({v3} de l'inertie totale)")
    if clusters is not None: 
        plt.legend()
    plt.show()


def display_corr_circle(pca, 
                        x_y, 
                        features) : 
    """
    Affiche le graphe des correlations

    Positional arguments : 
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    """

    # Extrait x et y 
    x,y=x_y

    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=(10, 9))

    # Pour chaque composante :
    for i in range(0, pca.components_.shape[1]):
        
        if (np.abs(pca.components_[x, i])<0.5 and not np.abs(pca.components_[y, i])<0.5):
            color = '#FFC300'
	elif (np.abs(pca.components_[y, i])<0.5 and not np.abs(pca.components_[x, i])<0.5):
            color = '#FFC300'
        elif (np.abs(pca.components_[x, i])<0.5 and np.abs(pca.components_[y, i])<0.5):
            color = '#FF0000'
        else:
            color = None
        # Les flèches
        ax.arrow(0,0, 
                pca.components_[x, i],  
                pca.components_[y, i],  
                head_width=0.07,
                head_length=0.07, 
                width=0.02,
                color=color)

        # Les labels
        plt.text(pca.components_[x, i] + np.sign(pca.components_[x, i])*0.06,
                pca.components_[y, i] + np.sign(pca.components_[y, i])*0.06,
                features[i])
        
    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel("F{} ({}%)".format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel("F{} ({}%)".format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    plt.title("Cercle des corrélations du plan (F{} , F{}) : {}% de l'inertie totale".format(x+1, y+1, 
                                                                                round(100*(pca.explained_variance_ratio_[x]+
                                                                                    pca.explained_variance_ratio_[y]),1)))

    # Le cercle 
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))

    # Axes et display
    plt.axis('equal')
    plt.show(block=False)