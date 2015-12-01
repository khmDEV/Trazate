import numpy as np
import matplotlib.pyplot as plt
import random as random
import time

def normalize(x):
    x /= np.linalg.norm(x)
    return x

def intersect_plane(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) == 0:
        return np.inf

    if denom<0:
        N=-N
        denom = np.dot(D, N)

    d = np.dot(P - O, N) / denom
    if d <= 0:
        return np.inf
    return d

def intersect_triangle(O, D, a, b, c, N):
    # Return the distance from O to the intersection of the ray (O, D) with the
    # triangle (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    d = intersect_plane(O,D,a,N);
    if d == np.inf:
        return np.inf
    p=O+D*d
    S1 = np.dot(np.cross((b-a),(p-a)),N)
    S2 = np.dot(np.cross((c-b),(p-b)),N)
    S3 = np.dot(np.cross((a-c),(p-c)),N)

    if S3<0:
        if S2<0:
            if S1<0:
                return d
    if S3>0:
        if S2>0:
            if S1>0:
                return d
    return np.inf

def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection of the ray (O, D) with the
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S

    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

def intersect(O, D, obj):
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])
    elif obj['type'] == 'triangle':
        return intersect_triangle(O, D, obj['a'], obj['b'], obj['c'], obj['normal'])

def get_normal(obj, M):
    # Find normal.
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    elif obj['type'] == 'triangle':
        N = obj['normal']
    return N

def get_color(obj, M):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color

def trace_ray(rayO, rayD):
    # Find first point of intersection with the scene.
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i


    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return
    # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    M = rayO + rayD * t
    # Find properties of the object.
    N = get_normal(obj, M)
    color = get_color(obj, M)
    toO = normalize(O - M)
    col_ray=0

    for i, l in enumerate(Lights):
        L=l["L"]
        color_light=l["color_light"]

        toL = normalize(L - M)
        dis=intersect(L, -toL, obj)
        # Shadow: find if the point is shadowed or not.
        lo = [intersect(M, toL, obj_sh)
                for k, obj_sh in enumerate(scene) if k != obj_idx]
        if not lo or min(lo) > dis:
            # Start computing the color.
            # Lambert shading (diffuse).
            col_ray += obj.get('diffuse_c', diffuse_c) * max(np.dot(N, toL), 0) * color
            # Blinn-Phong shading (specular).
            col_ray += obj.get('specular_c', specular_c) * max(np.dot(N, normalize(toL + toO)), 0) ** specular_k * color_light
    col_ray += color*ambient

    return obj, M, N, col_ray

def transform(matrix, obj):
    if obj['type'] == 'plane':
        transform_plane(matrix, obj)
    elif obj['type'] == 'sphere':
        transform_sphere(matrix, obj)
    elif obj['type'] == 'triangle':
        transform_triangle(matrix, obj)

def transform_triangle(matrix, triangle):
    triangle["a"]=transform_point(matrix,triangle["a"])
    triangle["b"]=transform_point(matrix,triangle["b"])
    triangle["c"]=transform_point(matrix,triangle["c"])
    triangle["normal"]=normalize(np.cross((triangle["b"]-triangle["a"]),(triangle["c"]-triangle["a"])))


def add_triangle(a, b, c, color):
    return dict(type='triangle', a=np.array(a),
        b=np.array(b), c=np.array(c),
        # ( (V2 - V1) x (V3 - V1) ) / | (V2 - V1) x (V3 - V1) |
        normal=normalize(np.cross((np.array(b)-np.array(a)),(np.array(c)-np.array(a))))
        , color=np.array(color),  reflection=.0, refraction=.0)

def transform_point(matrix, point):
    p=np.array(point[:])
    p=np.array(np.append(p,[1]))
    matrix=np.array(matrix)
    r=np.dot(p,matrix)
    return r[0:3]/r[3]

def transform_vector(matrix, point):
    p=np.array(point[:])
    p=np.array(np.append(p,[0]))
    matrix=np.array(matrix)
    return np.dot(p,matrix)[0:3]

def transform_sphere(matrix, sphere):
    sphere["position"]=transform_point(matrix,sphere["position"])

def add_sphere(position, radius, color):
    return dict(type='sphere', position=np.array(position),
        radius=np.array(radius), color=np.array(color), reflection=.0, refraction=0.9)

def transform_plane(matrix, plane):
    plane["position"]=transform_point(matrix,plane["position"])
    plane["normal"]=transform_vector(matrix,plane["normal"])

def add_plane(position, normal):
    return dict(type='plane', position=np.array(position),
        normal=np.array(normal),
        color=np.array([1.,1.,1.]),
        # color=lambda M: (color_plane1 if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane0),
        diffuse_c=.75, specular_c=.5, reflection=.0, refraction=0.)

def add_light(position,color):
    return dict(L = np.array(position), color_light = np.array(color))


def calculate_ray(rayO,rayD,max):
    col = np.zeros(3)
    col[:] = 0.
    colRfr = np.zeros(3)
    colRfr[:] = 0.
    colRefl = np.zeros(3)
    colRefl[:] = 0.
    if max==0:
        return []
    traced = trace_ray(rayO, rayD)
    if not traced:
        return []

    obj, M, N, col_ray = traced

    refraction = obj.get('refraction', 0.)
    reflection = obj.get('reflection', 0.)

    if reflection>1e-6:
        rayOref, rayDref = M + N * .0001, normalize(rayD - 2 * (np.dot(rayD, N)) * N)
        colRefl=calculate_ray(rayOref,rayDref,max-1)
        if colRefl==[]:
            reflection=0.
            colRefl=[0.,0.,0.]


    if np.abs(refraction)>1e-6:
        refractionN=refraction
        I=-rayD
        sc=np.dot(N,(I))
        if sc<0:
            N=-N
            sc=np.dot(N,(I))


        A=(refractionN*sc-np.sqrt(1-refractionN*refractionN*(1-sc*sc))) * N

        B=(refractionN*(I))
        rayDrefr = normalize(A - B) #np.array([0.,0.,1.])#
        rayOrefr = M + rayDrefr * .001

        colRfr=calculate_ray(rayOrefr,rayDrefr,max-1)
        if colRfr==[]:
            refraction=0.
            colRfr=[0.,0.,0.]

    col = col_ray*(1-refraction-reflection)+colRfr*refraction+colRefl*reflection


    return col


#Import .obj
def loadOBJ(filename):
    numVerts = 0
    verts = []
    norms = []
    vertsOut = []
    normsOut = []

    for line in open(filename, "r"):
        vals = line.split()
        if len(vals)>0:
            if vals[0] == "v":
                v = map(float, vals[1:4])
                verts.append(v)
            if vals[0] == "vn":
                n = map(float, vals[1:4])
                norms.append(n)
            if vals[0] == "f":
                for f in vals[1:]:
                    w = f.split("/")
                    # OBJ Files are 1-indexed so we must subtract 1 below
                    vertsOut.append(list(verts[int(w[0])-1]))
                    # normsOut.append(list(norms[int(w[2])-1]))
                    numVerts += 1
    return vertsOut, normsOut

n=3
antialiasing=3
w = 160*n
h = 90*n

# List of objects.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)
s=add_sphere([.75, .1, 4.], 1., [0., 0., 1.])
p=add_plane([0., 0, 10], [0., 0., -0.5])
t=add_triangle([2., 1., 2.], [2., 2., 3.], [-2., 3., 3.], [0., 0., 1.])

# T=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,2,0,1]])

# transform_triangle(T,t)
t1=add_triangle([0., 3., 2.5], [-0.5, 2., 3.], [0.5, 2., 3.], [0., 2., 1.])
t2=add_triangle([0., 3., 2.5], [-0.5, 2., 3.], [0., 2., 2.], [0., 2., 1.])
t3=add_triangle([0., 3., 2.5], [0.5, 2., 3.], [0., 2., 2.], [0., 2., 1.])
t4=add_triangle([-0.5, 2., 3.], [0.5, 2., 3.], [0., 2., 2.], [0., 2., 1.])

T=np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[-1,0,0,1]])
alf=0.3
R=np.array([[np.cos(alf),0,-np.sin(alf),0],[0,1,0,0],[np.sin(alf),0,np.cos(alf),0],[0,0,0,1]])
# transform_triangle(T,t1)
# transform_triangle(T,t2)
# transform_triangle(T,t3)
# transform_triangle(T,t4)

# transform_triangle(R,t1)
# transform_triangle(R,t2)
# transform_triangle(R,t3)
# transform_triangle(R,t4)
# s=add_sphere([2.75, .1, 2.25], .6, [.5, .223, .5])
s=add_sphere([0.75, .4, 2.5], .6, [.5, .223, .5])
s['reflection']=1.
s['refraction']=0.
s2=add_sphere([-0.75, .1, 2.5], .6, [1., .572, .184])
s2['reflection']=0.
s2['refraction']=0.95
scene = [
        # s,
        s,
        s2,
        add_plane([-5., 0., 0.], [1., 0., 0.]),
        add_plane([5., 0., 0.], [-1., 0., 0.]),
        add_plane([0., -1., 0.], [0., 1., 0.]),

        t1,
        t2,
        t3,
        t4,
        add_plane([0., 0, 10], [0., 0., -0.5]),
        add_sphere([2.75, .1, 4.], 1., [0., 0., 1.]),
        add_plane([0., 0, 10], [0., 0., -0.5]),
    ]

#Import .obj
listaVertices,listaNormales = loadOBJ("dodecaedro.obj")


it = iter(listaVertices)
# for x, y, z in zip(it, it, it):
#     scene.append(add_triangle(x,y,z,[.5, .223, .5]))
# print len(scene)

# Light position and color.
Lights=[
add_light([-0., 2., -0.],np.ones(3))
# add_light([0., 5., 0.],[0.,0.,1.])
# add_light([5., 5., 0.],[1.,0.,1.])
]
# T=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[10.,0.,0.,1.]])
# for i, obj in enumerate(scene):
#     transform(T,obj)

# Default light and material parameters.
ambient = .05
diffuse_c = 1.
specular_c = 1.
specular_k = 50.

depth_max = 5  # Maximum number of light reflections.
col = np.zeros(3)  # Current color.
O = np.array([0., 0, -1.])  # Camera.
Q = np.array([0., 0., 0.])  # Camera pointing to.



img = np.zeros((h, w, 3))

rh = float(w) / h
rw = float(h) / w

# Screen coordinates: x0, y0, x1, y1.
r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)
pixh = (S[1]-S[3]) / h
pixw = (S[0]-S[2]) / w

# Loop through all pixels.
arr=[[w] * h for i in range (w)]

white=np.ones(3)
start = time.time()

for i, x in enumerate(np.linspace(S[0], S[2], w)):
    if i % 10 == 0:
        print i / float(w) * 100, "%"
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col[:] = 0.
        n=0
        while n<antialiasing:
            xx=-pixh/2+random.random()*pixh
            yy=-pixw/2+random.random()*pixw
            # print "--"
            # print x
            # print xx
            # print y
            # print yy
            Q[:2] = (x, y)
            D = normalize(Q - O)
            depth = 0
            rayO, rayD = O, D
            rayV = 1.
            # Loop through initial and secondary rays.
            # while depth < depth_max:
            #     traced = trace_ray(rayO, rayD)
            #     if not traced:
            #         break
            #     obj, M, N, col_ray = traced
            #     # Reflection: create a new ray.
            #     rayO, rayD = M + N * .0001, normalize(rayD - 2 * (np.dot(rayD, N)) * N)
            #     depth += 1
            #     col += reflection * col_ray
            #     reflection *= obj.get('reflection', 1.)
            col+=calculate_ray(rayO,rayD,depth_max)
            # print "+++"
            # print n
            # print col
            n=n+1
        # img[h - j - 1, i, :] = np.clip(col/it, 0, 1)

        white=np.maximum(np.max(col),np.max(white))*np.ones(3)
        img[h - j - 1, i, :]=col
        # print "!!!!!!"
        # print img[h - j - 1, i, :]

for i, x in enumerate(np.linspace(S[0], S[2], w)):
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        # print "----"
        # print img[h - j - 1, i, :]
        # print np.clip(img[h - j - 1, i]/white, 0, 1)
        img[h - j - 1, i, :] = np.clip(img[h - j - 1, i]/white, 0, 1)

plt.imsave('fig.png', img)

end = time.time()
print end - start
