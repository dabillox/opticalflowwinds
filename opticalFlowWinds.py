'''
Created on 2 Aug 2013

@author: danielfisher
'''
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage as ndimage
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap 
import pyresample as pr
import cv2
import NPT.NonParametricTransform
import NPT.NonParametricMatcher
import mssl.cth
import mssl.m4utils

#
# Plotting
#

def draw_flow(im,flow,step=1):
    """ Plot optical flow at sample points
    spaced step pixels apart. """
    h,w = im.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T
    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)
    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,127),1)
        cv2.circle(vis,(x1,y1),1,(0,255,127), -1)
    return vis

def plot_gray(m,lat,lon,data,name):
    m.pcolormesh(lon, lat, data, latlon=True, cmap='gray')
    m.drawparallels(np.arange(-90,90,2),labels=[1,1,0,0])
    m.drawmeridians(np.arange(0,360,2),labels=[0,0,0,1])
    plt.savefig(name,bbox_inches='tight')
    plt.close()
    
def plot_color(m,lat,lon,data,name):
    m.pcolormesh(lon, lat, data, latlon=True)
    cbar = plt.colorbar()
    m.drawparallels(np.arange(-90,90,2),labels=[1,1,0,0])
    m.drawmeridians(np.arange(0,360,2),labels=[0,0,0,1])
    plt.savefig(name,bbox_inches='tight')
    plt.close()
    
def plot_winds(m, lat,lon, image, u, v, name, step=16):
    m.pcolormesh(lon, lat, image, latlon=True, cmap='gray')
    
    h,w = lat.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    
    #extract data
    u = u[y,x]
    v = v[y,x]
    lon = lon[y,x]
    lat = lat[y,x]
    
    #calculate map positionf lat lons
    x,y = m(lon,lat)
    
    # Calculate the orientation of the vectors
    x1, y1 = m(lon+u, lat+v)
    u_map, v_map = x1-x, y1-y
    
    # Rescale the magnitudes of the vectors...
    mag_scale = np.hypot(u_map, v_map) / np.hypot(u, v)
    u_map /= mag_scale
    v_map /= mag_scale
    
    # Draw barbs
    #m.barbs(x,y,u_map,v_map, length=5, color='red')
    m.quiver(x,y,u_map,v_map,scale=200, color='red')

    # Draw some grid lines for reference
    m.drawparallels(np.arange(-90,90,2),labels=[1,1,0,0])
    m.drawmeridians(np.arange(0,360,2),labels=[0,0,0,1])
    plt.savefig(name,bbox_inches='tight')
    plt.close()
    
def draw_maps(lat,lon,aatsr_eq,atsr2_eq,u,v, flow):
    
    #get extent
    minlat = np.min(lat)
    maxlat = np.max(lat)
    minlon = np.min(lon)
    maxlon = np.max(lon)
        
    #create map
    m = Basemap(projection='cyl', llcrnrlat=minlat-0.1, urcrnrlat=maxlat+0.1, llcrnrlon=minlon-0.1, urcrnrlon=maxlon+0.1,  resolution='l', area_thresh=1000)
    
    #do the plotting
    plot_winds(m, lat,lon, aatsr_eq, u, v, 'the_winds.png')
    plot_gray(m,lat,lon,aatsr_eq,'aatsr_eq.png')
    plot_gray(m,lat,lon,atsr2_eq,'atsr2_eq.png')
    plot_color(m,lat,lon,u,'u.png')
    plot_color(m,lat,lon,v,'v.png')
    plot_color(m,lat,lon,flow[:,:,0],'flow_x.png')
    plot_color(m,lat,lon,flow[:,:,1],'flow_y.png')
    
#
# Functions
#

def find_end_pixels(flow,dims):
    y,x = np.mgrid[1/2:dims[0]:1,1/2:dims[1]:1].reshape(2,-1)
    fx,fy = flow[y,x].T
    
    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)
    
    #exclude lines outside of image dimensions
    oob = lines >= np.max(dims)
    badRows = np.nonzero(oob.sum(axis=1) != 0)
    lines = np.delete(lines, badRows, axis=0) 
    oob = lines < 0
    badRows = np.nonzero(oob.sum(axis=1) != 0)
    lines = np.delete(lines, badRows, axis=0) 
    return lines
    
def shift_lat_lons(endPixels, lat, lon):
    shiftedLatGrid = np.zeros(lat.shape) 
    shiftedLonGrid = np.zeros(lat.shape)
    xLocs = endPixels[:,:,0]
    yLocs = endPixels[:,:,1]
    shiftedLatGrid[yLocs[:,0],xLocs[:,0]] = lat[yLocs[:,1],xLocs[:,1]]
    shiftedLonGrid[yLocs[:,0],xLocs[:,0]] = lon[yLocs[:,1],xLocs[:,1]]
    return shiftedLatGrid, shiftedLonGrid

def compute_distance_bearing(lat,lon,shiftedLat,shiftedLon):
    bearings = get_bearing(lat,lon,shiftedLat,shiftedLon)
    distances = get_distances_haversine(lat,lon,shiftedLat,shiftedLon)
    null = distances > 1000
    distances[null] = 0
    return distances, bearings

def get_bearing(lat,lon,shiftedLat,shiftedLon):
    lat =np.deg2rad(lat)
    lon = np.deg2rad(lon) 
    shiftedLat = np.deg2rad(shiftedLat)
    shiftedLon = np.deg2rad(shiftedLon)
    dLon = shiftedLon-lon
    y = np.sin(dLon)*np.cos(shiftedLat)
    x = np.cos(lat)*np.sin(shiftedLat) - np.sin(lat)*np.cos(shiftedLat)*np.cos(dLon)
    return np.arctan2(y,x) 

def get_distances_haversine(lat,lon,shiftedLat,shiftedLon):
    R = 6371
    dLat = np.deg2rad(shiftedLat-lat)
    dLon = np.deg2rad(shiftedLon-lon)
    lat1 = np.deg2rad(lat)
    lat2 = np.deg2rad(shiftedLat)
    a = np.sin(dLat/2) * np.sin(dLat/2) + \
          np.sin(dLon/2) * np.sin(dLon/2) * np.cos(lat1) * np.cos(lat2) 
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) 
    return R * c;

def get_distances_cosines(lat,lon,shiftedLat,shiftedLon):
    lat =np.deg2rad(lat)
    lon = np.deg2rad(lon) 
    shiftedLat = np.deg2rad(shiftedLat)
    shiftedLon = np.deg2rad(shiftedLon)
    R = 6371.0
    return np.arccos(np.sin(lat)*np.sin(shiftedLat) + np.cos(lat)*np.cos(shiftedLat)*np.cos(shiftedLon-lon)) * R

def get_speeds(distances):
    return (distances * 1000) / 1800

def convert_to_components(speeds, bearings):
    return -speeds * np.sin(bearings), -speeds * np.cos(bearings)
     

def hist_eq(im,nbr_bins=256):

    #get image histogram
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize

    #use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)

    return im2.reshape(im.shape).astype('uint8')
        
#
# Main script
#

#plot it
plot_it = True
write_out = False


#set result directory
resultDir = 'C:/Users/danielfisher/Data/WINDS/results/'

#set the nc files
data = nc.Dataset('C:/Users/danielfisher/Data/WINDS/netCDF/winds_final.nc','r')

#create the wind outputs
u_out = np.zeros(data.variables['btemp_nadir_1100_aatsr'][:,:].shape) + (-99999)
v_out = np.zeros(data.variables['btemp_nadir_1100_aatsr'][:,:].shape) + (-99999)
lon_out = np.zeros(data.variables['btemp_nadir_1100_aatsr'][:,:].shape) 
lat_out = np.zeros(data.variables['btemp_nadir_1100_aatsr'][:,:].shape) 
image_out = np.zeros(data.variables['btemp_nadir_1100_aatsr'][:,:].shape) 

sp= 1024
ep= sp + 512
#there are 85 block in the orbit used
for block in np.arange(77):
    print block
    
    #set the dir to hold maps
    if plot_it == True:
        imDir = resultDir +'/' + str(sp)
        try:
            os.mkdir(imDir)
        except:
            print 'dir exists'
        os.chdir(imDir)
    
    #get the current subset 
    try:
        aatsr = data.variables['btemp_nadir_1100_aatsr'][sp:ep,:]
        atsr2 = data.variables['btemp_nadir_1100_atsr2'][sp:ep,:] 
        lat = data.variables['latitude'][sp:ep,:]  
        lon = data.variables['longitude'][sp:ep,:]  
    except:
        sp = ep
        ep += 512
    
    #avoid the +- 180 dicontinuity
    if np.abs(np.min(lon) - np.max(lon)) > 180:
        print 'dateline crossing, moving to next tile...'
        sp = ep
        ep += 512
        continue
    
    #Histogram Eq. Data
    aatsr_eq = hist_eq(aatsr)
    atsr2_eq = hist_eq(atsr2)
      
    #get the optical flow using Farneback algorithm
    flow = cv2.calcOpticalFlowFarneback(atsr2_eq, aatsr_eq, 0.5, 4, 30, 7, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    #find flow end pixels
    endPixels = find_end_pixels(flow, lat.shape)
    
    #extract flow lat lons
    shiftedLat, shiftedLon = shift_lat_lons(endPixels, lat, lon)
    
    #compute distance and bearing
    distances, bearings = compute_distance_bearing(lat,lon,shiftedLat,shiftedLon)
    
    #convert distance to m/s
    speeds = get_speeds(distances)
    
    #convert to u and v components
    u, v = convert_to_components(speeds, bearings)
    
    #place into output array
    u_out[sp:ep,:] = u
    v_out[sp:ep,:] = v
    lon_out[sp:ep,:] = lon
    lat_out[sp:ep,:] = lat
    image_out[sp:ep,:] = aatsr
    
    #draw the maps
    if plot_it == True:
        try:
            draw_maps(lat,lon,aatsr_eq,atsr2_eq,u,v,flow)
        except:
            sp = ep
            ep += 512
            continue
    
    #continue the loop    
    sp = ep
    ep += 512
    
#write out netcdf
if write_out == True:
    ncfile = nc.Dataset('C:/Users/danielfisher/Desktop/WINDS/netCDF/atsr_tandem_winds.nc','w',format='NETCDF4')
    ncfile.createDimension('y', u_out.shape[0])
    ncfile.createDimension('x', u_out.shape[1])
    temp = ncfile.createVariable('u','f8',('y','x'), zlib='true')
    temp[:,:] = u_out
    temp = ncfile.createVariable('v','f8',('y','x'), zlib='true')
    temp[:,:] = v_out
    temp = ncfile.createVariable('heights','f8',('y','x'), zlib='true')
    temp[:,:] =  ndimage.median_filter(data.variables['Heights_ats_n'][:,:], 7)
    temp = ncfile.createVariable('lat','f8',('y','x'), zlib='true')
    temp[:,:] =lat_out
    temp = ncfile.createVariable('lon','f8',('y','x'), zlib='true')
    temp[:,:] = lon_out
    temp = ncfile.createVariable('image','f8',('y','x'), zlib='true')
    temp[:,:] = image_out
    temp = ncfile.createVariable('dem','f8',('y','x'), zlib='true')
    temp[:,:] = data.variables['dem'][:,:]
    ncfile.close()
    