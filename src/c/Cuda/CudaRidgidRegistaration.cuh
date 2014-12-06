#pragma once
#include "Vec.h"
#include "CudaUtilities.cuh"
#include "cuda_runtime.h"

// function normCoCube = iterateOverZ(maxIterZ,maxIterX,maxIterY,im1,im2,xStart1,xStart2,yStart1,yStart2,zStart1,zStart2,minOverlap,visualize)
// normCoCube = zeros(maxIterY*2,maxIterX*2,maxIterZ*2);
// 
// for delta = 1:maxIterZ*2
// curDelta = delta-maxIterZ;
// [start1,start2,end1,end2] = calculateROIs(curDelta,zStart1,zStart2,size(im1,3),size(im2,3));
// if(end1-start1<minOverlap/5 || end2-start2<minOverlap/5),continue,end
// 
// 	imZ1 = im1(:,: ,start1 : end1);
// imZ2 = im2(:,: ,start2 : end2);
// normCoCube(:,: ,delta) = iterateOverX(maxIterX,maxIterY,imZ1,imZ2,xStart1,xStart2,...
// 	yStart1,yStart2,curDelta,minOverlap,visualize);
// end
// imZ1 =[];
// imZ2 =[];
// end
// 
// function normCoSquare = iterateOverX(maxIterX,maxIterY,im1,im2,xStart1,xStart2,yStart1,yStart2,curDeltaZ,minOverlap,visualize)
// global Rect1 Rect2
// normCoSquare = zeros(maxIterY*2,maxIterX*2);
// 
// for delta = 1:maxIterX*2
// curDelta = delta-maxIterX;
// [start1,start2,end1,end2] = calculateROIs(curDelta,xStart1,xStart2,size(im1,2),size(im2,2));
// if(end1-start1<minOverlap || end2-start2<minOverlap),continue,end
// 
// 	if(visualize==1)
// 		pos1 = get(Rect1,'Position');
// pos2 = get(Rect2,'Position');
// set(Rect1,'Position',[max(start1,1),max(pos1(2),1),max(end1-start1,1),max(pos1(4),1)]);
// set(Rect2,'Position',[max(start2,1),max(pos2(2),1),max(end2-start2,1),max(pos2(4),1)]);
// end
// 
// imX1 = im1(:,start1 : end1,: );
// imX2 = im2(:,start2 : end2,: );
// normCoSquare(:,delta) = iterateOverY(maxIterY,imX1,imX2,curDelta,yStart1,yStart2,curDeltaZ,minOverlap,visualize);
// end
// imX1 =[];
// imX2 =[];
// end

// function normCoLine = iterateOverY(maxIterY,im1,im2,curDeltaX,yStart1,yStart2,curDeltaZ,minOverlap,visualize)
// global Rect1 Rect2
// normCoLine = zeros(maxIterY*2,1);
// 
// for delta = 1:maxIterY*2
// curDelta = delta-maxIterY;
// [start1,start2,end1,end2] = calculateROIs(curDelta,yStart1,yStart2,size(im1,1),size(im2,1));
// if(end1-start1<minOverlap || end2-start2<minOverlap),continue,end
// 
// 	imY1 = im1(start1:end1,: ,: );
// imY2 = im2(start2:end2,: ,: );
// 
// normCoLine(delta) = NormalizedCovariance(imY1,imY2);
// 
// if(visualize==1)
// pos1 = get(Rect1,'Position');
// pos2 = get(Rect2,'Position');
// set(Rect1,'Position',[max(pos1(1),1),max(start1,1),max(pos1(3),1),max(end1-start1,1)]);
// set(Rect2,'Position',[max(pos2(1),1),max(start2,1),max(pos2(3),1),max(end2-start2,1)]);
// updateXYviewer(imY1,imY2,normCoLine(delta),curDeltaX,curDelta,curDeltaZ);
// end
// 
// if(normCoLine(delta)>1 || normCoLine(delta)<-1)
// 	warning('Recived a NCV out of bounds:%f, overlap:(%d,%d,%d)',normCoLine(delta),size(imY1,2),size(imY1,1),size(imY1,3));
// normCoLine(delta) = 0;
// end
// end
// imY1 =[];
// imY2 =[];
// end
// 
// function[start1,start2,end1,end2] = calculateROIs(delta,oldStart1,oldStart2,size1,size2)
// if(oldStart1==1 && oldStart2~=1)
// start1 = 1;
// else
// start1 = max(1,oldStart1+delta);
// end
// 
// if(oldStart2==1 && oldStart1~=1)
// start2 = 1;
// else
// start2 = max(1,oldStart2-delta);
// end
// 
// minSize = min(size1-start1,size2-start2);
// end1 = start1 + minSize;
// end2 = start2 + minSize;
// 
// if(end1-start1~=end2-start2),error('Sizes dont`t match %d : %d!',end1-start1,end2-start2),end
// end

Vec<int> calcNewROI(Vec<int> delta,const Vec<int> starts1, const Vec<int> starts2,const Vec<size_t> imDim,
	Vec<int>& newStarts1, Vec<int>& newStarts2)
{
	newStarts1 = Vec<int>::max(Vec<int>(0,0,0),starts1+delta);
	newStarts2 = Vec<int>::max(Vec<int>(0,0,0),starts2+delta);
	return Vec<int>::min(imDim-starts1,imDim-starts2);
}

// Return pointer is to a new array that need to be cleaned up elsewhere!
template <class PixelType>
float* cRidgidRegistaration(PixelType* imageIn1, PixelType* imageIn1, Vec<size_t> imageDims, float** normCovarResults,
	Vec<size_t> maxIterations, Vec<int> roiStart1Init, Vec<int> roiStart2Init, Vec<int> minOverlap, int device=0)
{
	cudaSetDevice(device);

	float normCovarResults[] = new float[maxIterations.product()];
	for(int i=0; i<maxIterations.product(); ++i)
		normCovarResults[i] = 0;

	// Find the greatest overlap and consequently the largest image buffers needed.
	Vec<int> delta(0,0,0);
	Vec<size_t> maxSizes(0,0,0);
	size_t maxVol = 0;
	for(; delta.z<maxIterations.z; ++delta.z)
	{
		for(; delta.y<maxIterations.y; ++delta.y)
		{
			for(; delta.x<maxIterations.x; ++delta.x)
			{
				Vec<size_t> roiStart1,roiStart2;
				Vec<size_t> curSizes = Vec<size_t>(calcNewROI(delta,roiStart1Init,roiStart2Init,imageDims,roiStart1,roiStart2));
				if(maxVol<curSizes.product())
				{
					maxVol = curSizes.product();
					maxSizes = curSizes;
				}
			}
		}
	}


	return normCovarResults;
}