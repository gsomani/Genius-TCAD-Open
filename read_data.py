import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.tri as mtri
from matplotlib import ticker,cm
from scipy.interpolate import CubicSpline
import matplotlib.ticker as mticker

def log_tick_formatter(val, pos=None):
    return "{:.1e}".format(10**val)

def lexsort(a,points,dir=0):
    i = np.lexsort((points[:,1-dir],points[:,dir]))
    return np.array(a)[i]

def find_corners(edge,nodes):
    corners = np.zeros(len(nodes),dtype=int)
    for e in edge:
        n0,n1 = nodes[e[0]],nodes[e[1]]
        corners[e] ^= 1
    return  np.where(corners == 1)[0]

def decode_tif(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    points = []
    triangle = []
    edge = []
    interface = []
    contact = []
    var = []
    units = []
    sol = []
    material = []
    nodes = []
    region = []
    region_triangle = []
    for line in lines:
        l = line.split() 
        if(l[0]=='c'):
            points.append([l[2],l[3]])
        elif(l[0]=='t'):
            region_triangle.append(l[2])
            triangle.append(l[3:6])
        elif(l[0]=='e'):
            edge.append([int(l[2])-1,int(l[3])-1])
        elif(l[0]=='r'):
            region.append(l[2])
        elif(l[0]=='i'):
            contact.append(interface)
            interface = [l[3]] 
        elif(l[0]=='j'):
            interface.append(edge[int(l[1])-1])
        elif(l[0]=='s'):
            var = l[2:]
        elif(l[0]=='u'):
            units = l[2:]
        elif(l[0]=='n'):
            material.append(l[2])
            nodes.append(l[1]) 
            sol.append(l[3:])
    
    file.close()
    
    points = np.array(points,dtype=float) 
    triangle = np.array(triangle,dtype=int)-1
    region_triangle = np.array(region_triangle,dtype=int)-1
    tri = mtri.Triangulation(points[:,0],points[:,1],triangle)
   
    sol = np.array(sol,dtype=float)
    nodes = np.array(nodes,dtype=int)-1   
 
    contact.append(interface)
    contact = contact[1:]

    for i in range(len(contact)):
        c = contact[i]
        corners = find_corners(c[1:],points)
        contact[i] = [c[0],corners]
    
    solution = {}

    for i in range(len(var)):
        solution[var[i]] = units[i]

    materials = list(set(material))
    material = np.array(material)

    material_region = {}

    for j in range(len(materials)):
        m = materials[j]
        indices = np.where(material==m)[0]
        node = nodes[indices]
        solution[m] = {'nodes':node, 'triangles':[]}
        for i in range(len(var)):
            solution[m][var[i]] = sol[indices,i].tolist()

    for i in range(len(region)):
        r = region[i]
        t = np.where(region_triangle==i)[0]
        solution[r]['triangles'] += list(t)
     
    return tri,contact,materials,solution

def extract(var,triangles,sol):
    s = sol[var][1:]
    unit = sol[var][0]
    mask = triangles.mask
    triangles.set_mask(mask=None)
    if(unit=='cm^{-3}' and var != 'Net_Doping'):
        f = np.vectorize(mtri.LinearTriInterpolator(triangles,np.log(s)))
        def g(x,y):
            return np.exp(f(x,y))
        return s,g
    f = mtri.LinearTriInterpolator(triangles,s)
    triangles.set_mask(mask=mask)
    return np.array(s),np.vectorize(f)

def plot_structure(triangles,electrode,materials,solution,grid=True,linewidth=5,mesh=False,elec=True,color='0.7',xlabel='x',ylabel='y',title=True):
    Y = triangles.y
    X = triangles.x
    T = triangles.triangles
    plt.ylim([max(Y),min(Y)])
    unit = '($\mu m$)'
    plt.xlabel(xlabel + unit)
    plt.ylabel(ylabel + unit)
    
    color_material = ['orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','magenta', 'yellow']

    ones = np.ones(len(X))
    for i in range(len(materials)):
        m = materials[i]
        triangles.set_mask(mask(m,solution,triangles))
        plt.tricontourf(triangles,ones,colors=color_material[i])
    cmap = colors.ListedColormap(['blue']+color_material[:len(materials)]) 
    norm = plt.Normalize(-2,2*len(materials))     
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ticks=range(-1,2*len(materials),2))
    cbar.set_ticklabels(['Contact']+materials)

    if(elec):
        for e in electrode:
            b = e[1]
            x,y = X[b],Y[b] 
            plt.hlines(y[0],x[0],x[1],linewidth=linewidth,colors='blue')
 
    if(mesh):
        plt.triplot(triangles,color=color)
        plt.title("Mesh")
        plt.xlim([min(X),max(X)])
    
    if(mesh==False):    
        plt.grid(grid,'both')
    plt.gca().set_aspect('equal')
    
    plt.show()

def mask(material,solution,triangles):    
    tri = solution[material]['triangles']
    mask = np.ones(len(triangles.triangles),dtype=bool)
    mask[tri] = False
    return mask

def data_mask(data,sol,material,triangles):    
    nodes = sol[material]['nodes']
    mask = np.zeros(len(triangles.x))
    mask[nodes] = data
    return mask

def plot_sol_str(triangles,sol,material,var,dim=2,grid=True,vmin=None,vmax=None,levels=256,ni_level = 10,max_level = 21,cmap=cm.inferno,extend='neither',xlabel='x',ylabel='y',title=True):
    data = sol[material][var]
    data_unit = sol[var]
    plt.grid(grid,'both')
    plt.gca().set_aspect('equal')
    Y = triangles.y
    plt.ylim([max(Y),min(Y)])
    unit = '($\mu m$)'
    norm=None
    ticks=None
    display_var = var.replace('_',' ')
    
    triangles.set_mask(mask(material,solution,triangles))

    def extend_check(data,levels):
        cmp = [np.amin(data)-levels[0],np.amax(data)-levels[-1]]
        if(cmp[0]<0 and cmp[1]>0):
            extend='both'
        elif(cmp[1]>0):
            extend='max'
        elif(cmp[0]<0):
            extend='min'
        else:
            extend='neither'
        return extend

    if(var=='Net_Doping'):
        norm=colors.SymLogNorm(linthresh=1e9, linscale=1,base=10)
        lp = np.logspace(ni_level-1,max_level,levels+1)
        levels = np.concatenate([-lp[::-1],lp])
        tr = np.logspace(ni_level,max_level,max_level-ni_level+1)
        ticks = np.concatenate([-tr[::-1],[0],tr])
        extend=extend_check(data,levels)
    elif(unit=='cm^{-3}' and var!='Net charge carriers'):
        norm=colors.LogNorm()
        if(vmin != None and vmax != None):
            levels = np.logspace(vmin,vmax,levels)
            ticks = np.logspace(vmin,vmax,vmax-vmin+1)
        else:        
            levels = np.logspace(0,max_level,levels)
            ticks = np.logspace(0,max_level,max_level+1)
        extend=extend_check(data,levels)
    if(title==True):
        plt.title(display_var)
    if(dim==2):
        if(vmin != None and vmax != None):
            levels = np.linspace(vmin,vmax,levels)
            extend=extend_check(data,levels)
        plt.tricontourf(triangles, data_mask(data,solution,material,triangles),levels=levels,norm=norm,cmap=cmap,vmin=vmin,vmax=vmax,extend=extend)
        plt.colorbar(ticks=ticks).set_label("%s ($%s$)" %(display_var,data_unit))
        plt.xlabel(xlabel + unit)
        plt.ylabel(ylabel + unit)
        plt.gca().set_aspect('equal')
    else:
        if(unit=='cm^{-3}'):
            data = np.log10(data)
        plt.gca(projection='3d',proj_type = 'ortho').plot_trisurf(triangles, data_mask(data,solution,material,triangles), norm=norm)
        if(unit=='cm^{-3}'):
            plt.gca().set_zticks(np.log10(levels))
            plt.gca().zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        plt.gca().set_zlabel("%s ($%s$)" %(display_var,unit))        
    plt.show()
