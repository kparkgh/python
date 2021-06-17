# settings.py

#--------------------------------------------------------------
# os functions
#--------------------------------------------------------------

print('setting os functions...')
import os

chdir = os.chdir
listdir = os.listdir
getcwd = os.getcwd
remove = os.remove
rename = os.rename

#--------------------------------------------------------------
# sys functions
#--------------------------------------------------------------

print('setting sys functions...')
import sys

version = sys.version
path = sys.path

#--------------------------------------------------------------
# scipy.integrate functions
#--------------------------------------------------------------

print('setting scipy.integrate functions...')
from scipy.integrate import solve_ivp

#--------------------------------------------------------------
# scipy.interpolate functions
#--------------------------------------------------------------

print('setting scipy.interpolate functions...')
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import griddata

#--------------------------------------------------------------
# scipy.io functions
#--------------------------------------------------------------

print('setting scipy.io functions...')
import scipy.io as io

loadmat = io.loadmat

#--------------------------------------------------------------
# importlib functions
#--------------------------------------------------------------

print('setting importlib functions...')
import importlib

reload = importlib.reload

#--------------------------------------------------------------
# pandas and openpyxl functions
#--------------------------------------------------------------

print('setting pandas and openpyxl functions...')
import pandas as pd
import openpyxl

dataframe = pd.DataFrame
read_excel = pd.read_excel
excelwriter = pd.ExcelWriter

#--------------------------------------------------------------
# numpy functions
#--------------------------------------------------------------

print('setting numpy functions...')
import numpy as np

array = np.array
arange = np.arange

append = np.append
insert = np.insert
delete = np.delete
copy = np.copy

concatenate = np.concatenate
hstack = np.hstack
vstack = np.vstack

fromstring = np.fromstring

linspace = np.linspace
logspace = np.logspace

rand = np.random.rand
random = np.random.random
randint = np.random.randint
randn = np.random.randn
choice = np.random.choice
shuffle = np.random.shuffle

abs = np.abs
angle = np.angle
round = np.round
floor = np.floor
fix = np.fix
ceil = np.ceil
sign = np.sign

max = np.max  
min = np.min
prod = np.prod
mean = np.mean
std = np.std
sum = np.sum
diff = np.diff
cumsum = np.cumsum

argmax = np.argmax
argmin = np.argmin

unique = np.unique
sort = np.sort
searchsorted = np.searchsorted
reshape = np.reshape

meshgrid = np.meshgrid

shape = np.shape
size = np.size
where = np.where

zeros = np.zeros
ones = np.ones
full = np.full
eye = np.eye
diag = np.diag

sin = np.sin
cos = np.cos
tan = np.tan
exp = np.exp
log = np.log
sqrt = np.sqrt
arctan = np.arctan
arctan2 = np.arctan2

interp = np.interp
polyfit = np.polyfit
polyval = np.polyval

pi = np.pi
nan = np.nan
inf = np.inf
eps = np.finfo(float).eps
set_printoptions = np.set_printoptions

save = np.save
savez = np.savez
savetxt = np.savetxt
load = np.load
loadtxt = np.loadtxt

#--------------------------------------------------------------
# numpy.linalg functions
#--------------------------------------------------------------

print('setting numpy.linalg functions...')
import numpy.linalg

norm = numpy.linalg.norm
cond = numpy.linalg.cond
det = numpy.linalg.det

qr = numpy.linalg.qr
svd = numpy.linalg.svd
eig = numpy.linalg.eig

inv = numpy.linalg.inv
solve = numpy.linalg.solve
lstsq = numpy.linalg.lstsq

#--------------------------------------------------------------
# numpy.matlib functions
#--------------------------------------------------------------

print('setting numpy.matlib functions...')
import numpy.matlib

repmat = numpy.matlib.repmat

#--------------------------------------------------------------
# matplotlib.pyplot functions
#--------------------------------------------------------------

print('setting matplotlib.pyplot functions...')
import matplotlib.pyplot as plt

plt.ion() # interactive mode

plot = plt.plot
show = plt.show  # show()
draw = plt.draw  # draw()
subplot = plt.subplot
subplots = plt.subplots
figure = plt.figure
close = plt.close
title = plt.title
xlabel = plt.xlabel
ylabel = plt.ylabel
clabel = plt.clabel
text = plt.text  # text( x, y, 'string' )
grid = plt.grid  # grid(True)
xlim = plt.xlim  # plt.xlim( a, b )
ylim = plt.ylim  # plt.ylim( a, b )
legend = plt.legend  # plt.legend( ['string1' , 'string2', ...] )
axis = plt.axis
axes = plt.axes

ginput = plt.ginput

hist = plt.hist
hist2d = plt.hist2d
matshow = plt.matshow
colorbar = plt.colorbar
colormaps = plt.colormaps
set_cmap = plt.set_cmap

scatter = plt.scatter
contour = plt.contour
contourf = plt.contourf

imshow = plt.imshow

#--------------------------------------------------------------
# matplotlib.image functions
#--------------------------------------------------------------

print('setting matplotlib.image functions...')
import matplotlib.image as mpimg

imread = mpimg.imread

#--------------------------------------------------------------
# Axes3D functions
#--------------------------------------------------------------

#print('setting mpl_toolkits.mplot3d functions...')
#
# from mpl_toolkits.mplot3d import Axes3D

