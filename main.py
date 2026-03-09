import taichi as ti
import taichi.math as tm
import pygame as pg
import math as mt
import os
import numpy as np
ti.init(arch=ti.gpu,default_fp=ti.f64,fast_math=False)

PI = 3.1415926
z = 1.331
max_iter = 1000
real,imag = (-0.775,0)
complex_constant = [0,0]
res = [200,200]
frame = 0
focus = 0.35


# Functions for iteration
@ti.func
def vecmul(vec1,vec2):
    return tm.vec2(vec1[0]*vec2[0] - vec1[1]*vec2[1]  ,  vec1[0]*vec2[1] + vec1[1]*vec2[0])

@ti.func
def addvecs(vec1,vec2):
    return tm.vec2([vec1[0]+vec2[0],vec1[1]+vec2[1]])

@ti.func
def mandelbrot(vector,max_iter,c):
    z = tm.vec2([c[0],c[1]])
    iter = 0
    while  z[0]**2 + z[1]**2 < 4 and iter < max_iter:
        z = vecmul(z,z)
        z = addvecs(z,vector)
        iter +=1
    return iter   

@ti.func
def julia(vec,max_iter,c):
    v = tm.vec2(vec[0],vec[1])
    complex_number = tm.vec2(c[0],c[1])
    iter = 0
    while  v[0]**2 + v[1]**2 < 4 and iter < max_iter:
        v = vecmul(v,v)
        v = addvecs(v,complex_number)
        iter +=1
    return iter 

@ti.func
def col(value,threshold,u):
    r = (1-ti.floor(value/threshold))*((ti.sin(ti.f32(value*PI/180*6/2-2.75)))**2)
    g = (1-ti.floor(value/threshold))*(ti.sin(ti.f32(value*PI/180*5/2-2.75)))**2
    b = (1-ti.floor(value/threshold))*((0.76*ti.sin(ti.f32(value*PI/180*4/2-2.75)))**2+0.2)
    #r = (1/2+1/2*(ti.cos(ti.f32((value/threshold*2*PI)*0.9-(0.37*2*PI)))))*(1-ti.floor(value/threshold))
    #g = (1/2+1/2*(ti.cos(ti.f32((value/threshold*2*PI)*0.9-(PI**2)))))*(1-ti.floor(value/threshold))
    #b = (1/2+1/2*(ti.cos(ti.f32((value/threshold*2*PI)*0.8-(PI**2)))))*(1-ti.floor(value/threshold))
    return r,g,b
    
array = ti.Vector.field(3,dtype=float,shape=(res[0],res[1]))

#Main render function
@ti.kernel
def render(i:float,j:float,z:float,real:float,imag:float,focus:float,angle:float):
    
    for x,y in array:
            comp = tm.vec2(complex_constant[0]+i,complex_constant[1]+j)
            vector1 = tm.vec2(4*(x-res[0]/2)/res[0]/z+real,-(4*(y-res[1]/2)/res[0]/z)+imag)
            iter1 =mandelbrot(vector1,max_iter,comp)
            col1 = col(iter1,max_iter,angle)
            vector2 = tm.vec2(4*(x-res[0]/2)/res[0]/z+real,-(4*(y+focus-res[1]/2)/res[0]/z)+imag)
            iter2 =mandelbrot(vector2,max_iter,comp)
            col2 = col(iter2,max_iter,angle)
            vector3 = tm.vec2(4*(x-focus-res[0]/2)/res[0]/z+real,-(4*(y-res[1]/2)/res[0]/z)+imag)
            iter3 =mandelbrot(vector3,max_iter,comp)
            col3 = col(iter3,max_iter,angle)
            vector4 = tm.vec2(4*(x+focus-res[0]/2)/res[0]/z+real,-(4*(y-res[1]/2)/res[0]/z)+imag)
            iter4 =mandelbrot(vector4,max_iter,comp)
            col4 = col(iter4,max_iter,angle)
            vector5 = tm.vec2(4*(x-res[0]/2)/res[0]/z+real,-(4*(y-focus-res[1]/2)/res[0]/z)+imag)
            iter5 =mandelbrot(vector5,max_iter,comp)
            col5 = col(iter5,max_iter,angle)
            vector6 = tm.vec2(4*(x+focus-res[0]/2)/res[0]/z+real,-(4*(y-focus-res[1]/2)/res[0]/z)+imag)
            iter6 =mandelbrot(vector6,max_iter,comp)
            col6 = col(iter6,max_iter,angle)
            vector7 = tm.vec2(4*(x-focus-res[0]/2)/res[0]/z+real,-(4*(y+focus-res[1]/2)/res[0]/z)+imag)
            iter7 =mandelbrot(vector7,max_iter,comp)
            col7 = col(iter7,max_iter,angle)
            vector8 = tm.vec2(4*(x+focus-res[0]/2)/res[0]/z+real,-(4*(y+focus-res[1]/2)/res[0]/z)+imag)
            iter8 =mandelbrot(vector8,max_iter,comp)
            col8 = col(iter8,max_iter,angle)
            vector9 = tm.vec2(4*(x-focus-res[0]/2)/res[0]/z+real,-(4*(y-focus-res[1]/2)/res[0]/z)+imag)
            iter9 =mandelbrot(vector9,max_iter,comp)
            col9 = col(iter9,max_iter,angle)
            array[x,y][0] = (col1[0]+col2[0]+col3[0]+col4[0]+col5[0]+col6[0]+col7[0]+col8[0]+col9[0])/9
            array[x,y][1] = (col1[1]+col2[1]+col3[1]+col4[1]+col5[1]+col6[1]+col7[1]+col8[1]+col9[1])/9
            array[x,y][2] = (col1[2]+col2[2]+col3[2]+col4[2]+col5[2]+col6[2]+col7[2]+col8[2]+col9[2])/9
pg.init()

angle=0
i = -0.0
j = -0.0
u = 0


render(i,j,z,real,imag,focus,angle)
nparr = (array.to_numpy()*255).astype(int)
screensize = [900,900]
past_pos = [[z,i,j,real,imag,angle]]

gui = pg.display.set_mode((screensize[0],screensize[1]))
clock = pg.time.Clock()
running = True
while running:
    mousepos = pg.mouse.get_pos()
    cartesian_mouse_pos = (4*(mousepos[0]-screensize[0]/2)/screensize[0]/z+real,-(4*(mousepos[1]-screensize[1]/2)/screensize[0]/z)+imag)
    print("Mouse position:",cartesian_mouse_pos,f"Zoom: {z/(10**mt.floor(mt.log(z,10))):.5f}e{mt.floor(mt.log(z,10))}")
    fr = clock.get_fps()
    pg.display.set_caption(f"{clock.get_fps()}")
    past_pos.append([z,i,j,real,imag,angle])
    frame +=1
    keys = pg.key.get_pressed()
    if past_pos[0] != past_pos[1]:
        render(i,j,z,real,imag,focus,angle)
        nparr = (array.to_numpy()*255).astype(int)
    if len(past_pos) ==3:
        past_pos.pop(0)
    disp = pg.surfarray.make_surface(nparr)
    gui.fill("#000000")
    gui.blit(pg.transform.scale(disp,(screensize[0],(screensize[0]/screensize[1])*screensize[1])),(0,0))
    
    pg.draw.line(gui,"#ffffff39",(screensize[0]/2-5,screensize[1]/2),(screensize[0]/2+5,screensize[1]/2),2)
    pg.draw.line(gui,"#ffffff39",(screensize[0]/2,screensize[1]/2-5),(screensize[0]/2,screensize[1]/2+5),2)
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_r:
                z = 1
                i,j = 0,0
    pg.display.flip()
    if keys[pg.K_w] and fr > 0:
        imag +=0.09/z
    if keys[pg.K_s] and fr > 0:
        imag -=0.09/z
    if keys[pg.K_a] and fr > 0:
        real -=0.09/z
    if keys[pg.K_d] and fr > 0:
        real +=0.09/z
    if keys[pg.K_UP] and fr > 0:
        j +=0.05/z
        points = pointmandelbrot(cartesian_mouse_pos,max_iter,[complex_constant[0]+i,complex_constant[1]+j])
    if keys[pg.K_DOWN] and fr > 0:
        j -=0.05/z
        points = pointmandelbrot(cartesian_mouse_pos,max_iter,[complex_constant[0]+i,complex_constant[1]+j])
    if keys[pg.K_LEFT] and fr > 0:
        i -=0.05/z
        points = pointmandelbrot(cartesian_mouse_pos,max_iter,[complex_constant[0]+i,complex_constant[1]+j])
    if keys[pg.K_RIGHT] and fr > 0:
        i +=0.05/z
        points = pointmandelbrot(cartesian_mouse_pos,max_iter,[complex_constant[0]+i,complex_constant[1]+j])
    if keys[pg.K_f] and fr > 0:
        z*=1.1
    if keys[pg.K_b] and fr > 0:
        z /=1.1
    clock.tick(60)
