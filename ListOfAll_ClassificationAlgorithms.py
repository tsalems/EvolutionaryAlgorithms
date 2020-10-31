# list of all classification algorithms from sklearn

from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin

classifiers=[est for est in all_estimators() if issubclass(est[1], ClassifierMixin)]
# print(classifiers)
for clf in zip(classifiers):
    print(clf)

estimators = all_estimators(type_filter='classifier')

print("\n") #############################################################################################

all_clfs = []
for name, ClassifierClass in estimators:
    print('Appending', name)
    clf = ClassifierClass()
    all_clfs.append(clf)