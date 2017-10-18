import cv2
import math
import numpy as np
import pickle
import pygame

drawing = False # true if mouse is pressed
ix,iy = -1,-1
lastx, lasty = 0, 0
thetaNow, phiNow = 0, 0
center = 0
planeW = 950
planeH = 616
img = cv2.imread('ndhulib4.jpg')
img = img[:, :, ::-1]
y = np.arange(-planeH // 2, planeH // 2)
x = np.arange(-planeW // 2, planeW // 2)
xv, yyv = np.meshgrid(x, y, sparse=False)
oimg = None


def buildTable(img, scale, zoom):
    global x, y, xv, yyv
    print('Z = ', zoom)
    height, width = img.shape[0], img.shape[1]
    R = width / 4 * zoom
    # z = R
    zz = []
    uuu = []
    vvv = []
    for phi in range(0, 91):
        print(phi)
        phi = phi * math.pi / 180  # 0.95*math.pi/5
        yv = yyv * math.cos(phi) - R * math.sin(phi)
        z = yyv * math.sin(phi) + R * math.cos(phi)
        z = np.round(z)
        t1 = z
        t1[t1 == 0] = 1.0
        t1 = xv / t1
        t1[z == 0] = np.inf
        t1 = np.arctan(t1)
        t2 = xv ** 2 + z ** 2
        t2[t2 == 0] = 1.0
        t2 = yv / t2 ** 0.5
        t2[xv ** 2 + z ** 2 == 0] = np.inf
        t2 = np.arctan(t2)
        u = np.floor(width * t1 / (2 * math.pi))  # np.floor(R*xv/(xv**2+yv**2+z**2)**0.5+width/2)
        if scale != 1.0:
            u = cv2.resize(u, None, fx=scale, fy=scale)
        uu = np.reshape(u, (-1,))
        uuu.append(uu.astype(np.int16))
        # u[z<0] = u[z<0] + width/2 #*****
        # u[u>=width] = u[u>=width] % width #*****
        v = np.floor(height * t2 / (math.pi) + height / 2)  # np.floor(R*yv/(xv**2+yv**2+z**2)**0.5+height/2)
        v[v >= height] = height - 1
        if scale != 1.0:
            v = cv2.resize(v, None, fx=scale, fy=scale)
        vv = np.reshape(v, (-1,))
        vvv.append(vv.astype(np.int16))
        if scale != 1.0:
            z = cv2.resize(z, None, fx=scale, fy=scale)
        zz.append(np.reshape(z.astype(np.int16), (-1,)))
    pickle.dump([zz, uuu, vvv], open('table%dx%d_s%f_i%dx%d.dat' % (planeW, planeH, scale, width, height), 'wb'))
    return zz, uuu, vvv


def readTable(planeW, planeH, scale, width, height):
    z, uuu, vvv = pickle.load(open('table%dx%d_s%f_i%dx%d.dat' % (planeW, planeH, scale, width, height), 'rb'))
    for t in range(0, 91):
        uuu[t] = np.reshape(uuu[t], (round(planeH * scale), round(planeW * scale)))
        uuu[t] = cv2.resize(uuu[t], None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_NEAREST)
        uuu[t] = np.reshape(uuu[t], (-1,))
        vvv[t] = np.reshape(vvv[t], (round(planeH * scale), round(planeW * scale)))
        vvv[t] = cv2.resize(vvv[t], None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_NEAREST)
        vvv[t] = np.reshape(vvv[t], (-1,))
        z[t] = np.reshape(z[t], (round(planeH * scale), round(planeW * scale)))
        z[t] = cv2.resize(z[t], None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_NEAREST)
        z[t] = np.reshape(z[t], (-1,))
    return z, uuu, vvv


def unwarpFast(img, theta, phi, zoom, planeW, planeH):
    global z, uuu, vvv
    height, width = img.shape[0], img.shape[1]
    theta = theta * math.pi / 180  # math.pi
    center = height / math.pi * theta
    #     phi += 90
    if phi >= 0:
        uuuu = uuu[phi]
        zzzz = z[phi]
        vvvv = vvv[phi]
    else:
        uuuu = uuu[-phi]
        uuuu = -uuuu[::-1]
        # vvvv = vvv[-phi]
        zzzz = z[-phi]
        zzzz = zzzz[::-1]
        vvvv = img.shape[0] - vvv[-phi]
        vvvv = vvvv[::-1]
    u = np.floor(uuuu + center)  # np.floor(R*xv/(xv**2+yv**2+z**2)**0.5+width/2)
    u[zzzz < 0] = u[zzzz < 0] + width / 2
    u[u >= width] = u[u >= width] % width
    v = vvvv
    v[v >= height] = height - 1
    oimg = img[v, u.astype(np.int16)]
    # print(planeH*planeW*3, oimg.shape)
    oimg = np.reshape(oimg, (planeH, planeW, 3))
    return oimg


def unwarp(img, theta, phi, zoom, planeW, planeH):
    global x, y, xv, yyv
    oimg = np.zeros((planeH, planeW), dtype=np.uint8)
    height, width = img.shape[0], img.shape[1]
    R = width / 4 * zoom
    z = R
    #     y = np.arange(-planeH//2,planeH//2)
    #     x = np.arange(-planeW//2,planeW//2)
    #     xv, yyv = np.meshgrid(x, y, sparse=False)
    phi = phi * math.pi / 180  # 0.95*math.pi/5
    yv = yyv * math.cos(phi) - z * math.sin(phi)
    z = yyv * math.sin(phi) + z * math.cos(phi)
    t1 = z
    t1[t1 == 0] = 1.0
    t1 = xv / t1
    t1[z == 0] = np.inf
    t1 = np.arctan(t1)
    t2 = xv ** 2 + z ** 2
    t2[t2 == 0] = 1.0
    t2 = yv / t2 ** 0.5
    t2[xv ** 2 + z ** 2 == 0] = np.inf
    t2 = np.arctan(t2)
    theta = theta * math.pi / 180  # math.pi
    center = height / math.pi * theta
    u = np.floor(width * t1 / (2 * math.pi) + center)  # np.floor(R*xv/(xv**2+yv**2+z**2)**0.5+width/2)
    u[z < 0] = u[z < 0] + width / 2
    u[u >= width] = u[u >= width] % width
    v = np.floor(height * t2 / (math.pi) + height / 2)  # np.floor(R*yv/(xv**2+yv**2+z**2)**0.5+height/2)
    v[v >= height] = height - 1
    uu = np.reshape(u, (-1,))
    uu = uu.astype(np.int32)
    vv = np.reshape(v, (-1,))
    vv = vv.astype(np.int32)
    oimg = img[vv, uu]
    oimg = np.reshape(oimg, (planeH, planeW, 3))
    return oimg

def unwarpFast2(img, theta, phi, zoom, planeW, planeH, center):
    global z, uuu, vvv
    height, width = img.shape[0], img.shape[1]
    theta = theta * math.pi / 180  # math.pi
    # center = height / math.pi * theta
    #     phi += 90
    if phi >= 0:
        uuuu = uuu[phi]
        zzzz = z[phi]
        vvvv = vvv[phi]
    else:
        uuuu = uuu[-phi]
        uuuu = -uuuu[::-1]
        # vvvv = vvv[-phi]
        zzzz = z[-phi]
        zzzz = zzzz[::-1]
        vvvv = img.shape[0] - vvv[-phi]
        vvvv = vvvv[::-1]
    #print('center=', center)
    u = np.floor(uuuu + center)  # np.floor(R*xv/(xv**2+yv**2+z**2)**0.5+width/2)
    u[zzzz < 0] = u[zzzz < 0] + width / 2
    u[u >= width] = u[u >= width] % width
    v = vvvv
    v[v >= height] = height - 1
    oimg = img[v, u.astype(np.int16)]
    # print(planeH*planeW*3, oimg.shape)
    oimg = np.reshape(oimg, (planeH, planeW, 3))
    return oimg

# def drawPanorama(event,x,y,flags,param):
def drawPanorama():
    global ix, iy, drawing, thetaNow, phiNow, oimg, center, lastx, lasty
    exit = False
    x, y = pygame.mouse.get_pos()
    if x==lastx and y==lasty:
        ix, iy = x, y
    else:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
                ix,iy = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEMOTION:
                x, y = pygame.mouse.get_pos()
                if drawing:
                    speedx, speedy = 3*round((x-ix)), round((y-iy)/2)
                    center -= speedx
                    center %= img.shape[1]
                    phiNow += speedy
                    #phiNow = np.sign(phiNow)*(abs(phiNow)%90)
                    if phiNow>90:
                        phiNow = 90
                    elif phiNow<-90:
                        phiNow = -90
                    oimg = unwarpFast2(img, thetaNow, phiNow, 1, planeW, planeH, center)
                    # print(x, ix, speedx, speedy)
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    exit = True
            ix,iy = x,y
    return exit

#setting up pygame
def run():
    global z, uuu, vvv, oimg
    scale = 1.0
    zoom = 0.2
    # z, uuu, vvv = buildTable(img, scale, zoom=zoom)
    z, uuu, vvv = readTable(planeW, planeH, scale, img.shape[1], img.shape[0])
    pygame.init()
    screen = pygame.display.set_mode((planeW,planeH))
    pygame.display.set_caption('Alien Invasion') #設定視窗標題
    # cv2.namedWindow('Panorama')
    # cv2.setMouseCallback('Panorama', drawPanorama)
    oimg = unwarpFast2(img, thetaNow, phiNow, 1, planeW, planeH, center)
    exit = False
    while not exit:
        if oimg is not None:
            # cv2.imshow('Panorama', oimg)
            exit = drawPanorama()
            surf = pygame.surfarray.make_surface(oimg)
            surf = pygame.transform.flip(surf, False, True)
            surf = pygame.transform.rotate(surf, -90)
            screen.blit(surf, (0, 0))
            pygame.display.update()
        # k = cv2.waitKey(1) & 0xFF
        # if k == 27:
        #     break
    pygame.quit()
    # cv2.destroyAllWindows()

    #pygame.init()
    #screen = pygame.display.set_mode((planeW,planeH))
    #pygame.display.set_caption('Alien Invasion') #設定視窗標題
    # for t in np.arange(0, 360, 2):
    #     oimg = unwarpFast(img, t, 0, 1, planeW, planeH)
    #     cv2.imshow('out', oimg)

        # surf = pygame.surfarray.make_surface(oimg)
        # surf = pygame.transform.flip(surf, False, True)
        # surf = pygame.transform.rotate(surf, -90)
        # screen.blit(surf, (0, 0))
        # pygame.display.update()
        #pygame.display.flip()
    # pygame.quit()
    # cv2.destroyAllWindows()

run()

