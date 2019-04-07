
from PIL import Image
import numpy as np

class pxy: # +x is right and +y is up
    #for pxy objects a png file is associated with a numpy array with methods to draw lines and fill polygons
    red = [120,20,20] ; green = [20, 100, 20]; blue = [20, 60, 120]; white = [255, 255, 255]; light = [160, 160, 160]; dark = [50, 50,50]; grey = [110,110,110]

    def __init__(self, xxx, yyy):  
        self.xsize = xxx ; self.ysize = yyy # and ysize is also used in MAIN to de-invert y axis
        self.np_im = np.zeros((self.ysize, self.xsize,3), dtype=np.uint8) #set up a page array with three colours
        self.np_im [:,:] = pxy.dark; self.penrgb = pxy.white; self.filename = "page1.png" #defaults eg page colour

    def save(self):    
        Image.fromarray(self.np_im).save(self.filename)

    def horizline(self,yy1,xx1,xx2):
        for ixx in range(xx1,xx2,1-2*(xx2<xx1)): self.np_im[yy1,ixx] = self.penrgb

    def polyfiller(self): # using self.unn array of 2D points on U & V axes as polygon for filling
        pu = self.unn; P=np.shape(pu)[0]; ips=pu[::,1].argsort(); ib=ips[0]; it=ips[P-1]
        ms=np.zeros((P,2),dtype='int'); ns=np.zeros((P,2),dtype='int')
        j=ib; k=ib; m=0; n=0; ms[m]=pu[j];ns[n]=pu[k] 
        while j!=it: m+=1; j=j+1-P*(j==(P-1)); ms[m]=pu[j] #m[] gets anticlock (as pu) points from b to t
        while k!=it: n+=1; k=k-1+P*(k==0); ns[n]=pu[k] #n[] gets clockwise (opp to pu) points from b to t
        m=1; n=1; Sy=ms[0,1]; Sx=ms[0,0]; Ex=ms[0,0]# m,n start again now so m,n,S,E from bottom of poly
        while m+n<=P: #move S & E up M & N poly legs until (while not at) top of poly, plotting horizline as you go
            dxm=0; dxn=0; dm=(ms[m,1]-ms[m-1,1]); dn=(ns[n,1]-ns[n-1,1])
            if dm!=0: dxm=(ms[m,0]-ms[m-1,0])/dm
            if dn!=0: dxn=(ns[n,0]-ns[n-1,0])/dn
            if ms[m,1]<=ns[n,1]: 
                while Sy<ms[m,1]: Sy+=1; Sx+=dxm; Ex+=dxn; p2.horizline(Sy,int(Sx),int(Ex))
                Sx=ms[m,0]; m+=1 #Sx should be there already but maybe with cumulated rounding errors
            else: 
                while Sy<ns[n,1]: Sy+=1; Sx+=dxm; Ex+=dxn; p2.horizline(Sy,int(Sx),int(Ex))
                Ex=ns[n,0]; n+=1 # Ex note as Sx above

#MAIN
Eye = np.array([15,-10,30]) #view/camera point for perspective CHANGE THIS FOR DIFFERENT VIEW eg ([15,-10,30]) 
bStraightenVerts = False
bRandTinge = True; bThisRefDir = True; RefDir = [3,-20,1] # pointing toward (sic) illum source, any length - ignored if bThisRefDir is False

#Create the object p2 as an instance of the pxy class defined above
p2 = pxy(800, 800); p2.filename = 'yPdraw1.png'    

#POINTS AND FACES - faces of 3,4,5 vertices/sides (or more) have all points specified counterclockwise from outside 
delProjpt = np.array([0,0,0]); uvscale = 40; ushift = 0; vshift = 0 #change proj point, screen scale or screen coords
B = np.array([[0,0,0],[5,0,0],[5,5,0],[0,5,0],[0,0,3],[5,0,3],[5,5,3],[0,5,3],[0,8,0],[5,8,0],[5,13,0],[0,13,0],[0,8,4],[5,8,2],[5,13,2],[0,13,4],[2.5,0,4.5],[2.5,3.5,4.5]])
F3 = np.array([[6,7,17]]) # or could be np.empty((0,3),int)
F4 = np.array([[2,1,0,3],[1,2,6,5],[2,3,7,6],[3,0,4,7],[10,9,8,11],[8,9,13,12],[9,10,14,13],[10,11,15,14],[11,8,12,15],[12,13,14,15],[5,6,17,16],[7,4,16,17]])
F5 = np.array([[0,1,5,16,4]]) # eg np.empty((0,5),int) or np.array([[0,1,5,16,4],[2,3,7,17,6]])
#Load all face specs into a single numpy array, always starting with the number of sides/vertices
Fmax=5; nF3=np.shape(F3)[0]; nF4=np.shape(F4)[0]; nF5=np.shape(F5)[0]; nF=nF3+nF4+nF5
F=np.zeros((nF,1+Fmax),int); f=0
for i in range(nF3): F[f,0]=3; F[f,1:4]=F3[i]; f+=1 
for i in range(nF4): F[f,0]=4; F[f,1:5]=F4[i]; f+=1 
for i in range(nF5): F[f,0]=5; F[f,1:6]=F5[i]; f+=1 

#SET UP PROJECTION
nB=np.shape(B)[0]; midB = (B.min(axis=0) + B.max(axis=0))/2; Projpoint = midB + delProjpt #delProject normally zero
if bStraightenVerts: Projpoint[2] = Eye[2] ; vshift += Eye[2] #hmmm seems to work
EP = Projpoint - Eye ; EPx = EP[0] ; EPy = EP[1]; EPz = EP[2]
U = np.array([EPy,-EPx,0]) # this is EPcross(unit +z axis) so U is rightward horizontal in projection plane 
V = np.array([-EPx*EPz, -EPy*EPz, EPy*EPy + EPx*EPx]) #this is UcrossEP so V is upward in projection plane
if EPx==0 and EPy==0: U = [1,0,0]; V = [0,1,0] # if directly above projpoint set U and V as +x and +y
U = U/np.linalg.norm(U) ; V = V/np.linalg.norm(V) #... & now unit length
A = np.zeros((3,3)); Bp = np.zeros((nB,2))
#Project each 3D point B to 2D point Bp in projection plane NB Y axis is de-inverted
for b in range(nB):
    A = [B[b] - Eye , U , V]; Ainv = np.asarray(np.mat(A).I); res = np.dot(EP,Ainv)
    Bp[b] = [round(p2.xsize/2 + uvscale*(ushift-res[1])), round(p2.ysize/2 - uvscale*(vshift-res[2]))]
    
#FILL PERSP FACES
#Sort by distance - for each face record in array D[] the distance of mid point from Eye and get sort-indices
D = np.zeros(nF)
for f in range(nF):
    mid = (B[F[f,1:F[f,0]+1]].min(axis=0) + B[F[f,1:F[f,0]+1]].max(axis=0))/2 
    D[f] = np.linalg.norm(mid - Eye) # distance from eye
I = np.argsort(D)[::-1] # reversed so descending list... and of INDEX (ie argument) not values
#Fill vertices of each projected poly - most distant face first, closest last
for f in range(nF): # get s is next face starting from furthest
    s = I[f]; nSides = F[s,0]; p2.unn = np.zeros((nSides,2),dtype='int')
    #Load projected points for this face into p2.unn
    for i in range(nSides): p2.unn[i] = Bp[F[[s],i+1]] # so p2.unn has the projected points of the poly 
    #Find shading according to face aligned with view direction or RefDir specified at start
    normS = np.cross(B[F[s,2]]-B[F[s,1]],B[F[s,3]]-B[F[s,2]]); normS = normS/np.linalg.norm(normS) #unit outward normal to face
    if not bThisRefDir: RefDir = Eye - midB # otherwise it's as initially set at start
    RefDir = RefDir/np.linalg.norm(RefDir); Lnorm = (0.5*np.dot(normS,RefDir)+0.5) 
    shade = 80 + 120*Lnorm; smult = 30*bRandTinge # so faces with normal closer to RefDir are brighter
    #Fill
    p2.penrgb = [shade-smult*np.random.random(1), shade-smult*np.random.random(1), shade-smult*np.random.random(1)]
    p2.polyfiller() #using p2.unn array of nSides x 2D as created in this loop 
p2.save() #makes and saves png image from 2D rgb-valued np array

print('Eye',Eye,'   Projection point',Projpoint, '   Straightened', bStraightenVerts,'\nEnd\n')    