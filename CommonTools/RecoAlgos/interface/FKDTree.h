#ifndef FKDTREE_FKDTREE_H_
#define FKDTREE_FKDTREE_H_

#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <utility>
#include "FKDPoint.h"
#include "FQueue.h"


namespace{
    const std::array<unsigned int,32> MultiplyDeBruijnBitPosition{{0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
  8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31}};
      unsigned int ilog2 (unsigned int v)
    {
    v |= v >> 1; // first round down to one less than a power of 2 
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return MultiplyDeBruijnBitPosition[(unsigned int)(v * 0x07C4ACDDU) >> 27];

    }


/*    unsigned int ilog2 (unsigned int v)
    {
       return (unsigned int)floor(std::log2(v));
    
    }
    */
}




template<class TYPE, unsigned int numberOfDimensions>
class FKDTree
{

public:

	FKDTree(const unsigned int nPoints)
	{
		theNumberOfPoints = nPoints;

		theDepth = ilog2(nPoints);
		for (auto& x : theDimensions)
			x.resize(theNumberOfPoints);
		theIntervalLength.resize(theNumberOfPoints, 0);
		theIntervalMin.resize(theNumberOfPoints, 0);
		theIds.resize(theNumberOfPoints);
		thePoints.reserve(theNumberOfPoints);

	}




	FKDTree()
	{
		theNumberOfPoints = 0;
		theDepth = 0;
		for (auto& x : theDimensions)
			x.clear();
		theIntervalLength.clear();
		theIntervalMin.clear();
		theIds.clear();
		thePoints.clear();
	}

	FKDTree(const FKDTree<TYPE, numberOfDimensions>& v);

	FKDTree(FKDTree<TYPE, numberOfDimensions> && other)
	{
		theNumberOfPoints(std::move(other.theNumberOfPoints));
		theDepth(std::move(other.theDepth));

		theIntervalLength.clear();
		theIntervalMin.clear();
		theIds.clear();
		thePoints.clear();
		for (auto& x : theDimensions)
			x.clear();

		theIntervalLength = std::move(other.theIntervalLength);
		theIntervalMin = std::move(other.theIntervalMin);
		theIds = std::move(other.theIds);

		thePoints = std::move(other.thePoints);
		for (unsigned int i = 0; i < numberOfDimensions; ++i)
			theDimensions = std::move(other.theDimensions);
	}

	FKDTree<TYPE, numberOfDimensions>& operator=(
			FKDTree<TYPE, numberOfDimensions> && other)
	{

		if (this != &other)
		{
			theNumberOfPoints(std::move(other.theNumberOfPoints));
			theDepth(std::move(other.theDepth));

			theIntervalLength.clear();
			theIntervalMin.clear();
			theIds.clear();
			thePoints.clear();
			for (auto& x : theDimensions)
				x.clear();

			theIntervalLength = std::move(other.theIntervalLength);
			theIntervalMin = std::move(other.theIntervalMin);
			theIds = std::move(other.theIds);

			thePoints = std::move(other.thePoints);
			for (unsigned int i = 0; i < numberOfDimensions; ++i)
				theDimensions = std::move(other.theDimensions);
		}
		return *this;

	}

	void push_back(const FKDPoint<TYPE, numberOfDimensions>& point)
	{

		thePoints.push_back(point);
		for (unsigned int i = 0; i < numberOfDimensions; ++i)
			theDimensions.at(i).push_back(point[i]);
		theIds.push_back(point.getId());

	
	}

	void push_back(FKDPoint<TYPE, numberOfDimensions> && point)
	{

		thePoints.push_back(point);
		for (unsigned int i = 0; i < numberOfDimensions; ++i)
			theDimensions.at(i).push_back(point[i]);
		theIds.push_back(point.getId());

	}





	std::vector<unsigned int> search(
			const FKDPoint<TYPE, numberOfDimensions>& minPoint,
			const FKDPoint<TYPE, numberOfDimensions>& maxPoint,
			std::vector<unsigned int>& foundPoints, bool debug=false)
	{
		FQueue<unsigned int> indicesToVisit(16);
		indicesToVisit.push_back(0);
		unsigned int index;
		bool intersection;
		int dimension;
		unsigned int numberOfindicesToVisitThisDepth;
		int numberOfSonsToVisitNext;
		unsigned int firstSonToVisitNext;

		for (unsigned int depth = 0; depth < theDepth + 1; ++depth)
		{

			dimension = depth % numberOfDimensions;
			numberOfindicesToVisitThisDepth = indicesToVisit.size();
			for (unsigned int visitedindicesThisDepth = 0;
					visitedindicesThisDepth < numberOfindicesToVisitThisDepth;
					visitedindicesThisDepth++)
			{

				index = indicesToVisit[visitedindicesThisDepth];
				intersection = intersects(index, minPoint, maxPoint, dimension);
				firstSonToVisitNext = leftSonIndex(index);
            

				if (intersection)
				{
					if (is_in_the_box(index, minPoint, maxPoint))
					{
										
						foundPoints.emplace_back(theIds[index]);
					}
					numberOfSonsToVisitNext = (firstSonToVisitNext
                  < theNumberOfPoints)
                  + ((firstSonToVisitNext + 1) < theNumberOfPoints);
				}
				else
				{


					firstSonToVisitNext += (theDimensions[dimension][index]
							< minPoint[dimension]);

               numberOfSonsToVisitNext = std::min((firstSonToVisitNext
                  < theNumberOfPoints)
                  + ((firstSonToVisitNext + 1) < theNumberOfPoints) 
                     , 1);
				}

				for (int whichSon = 0; whichSon < numberOfSonsToVisitNext;
						++whichSon){
               
					indicesToVisit.push_back(firstSonToVisitNext + whichSon);
			      }
			}

			indicesToVisit.pop_front(numberOfindicesToVisitThisDepth);
		}
		return foundPoints;
	}


	void build()
	{
		//this happens if the kdtree has been filled one point per time
		if(theNumberOfPoints != thePoints.size())
		{
			theNumberOfPoints =thePoints.size();
			theDepth = ilog2(theNumberOfPoints);
			theIntervalLength.resize(theNumberOfPoints, 0);
			theIntervalMin.resize(theNumberOfPoints, 0);
		}

		//gather kdtree building
		int dimension;
		theIntervalMin[0] = 0;
		theIntervalLength[0] = theNumberOfPoints;

		for (unsigned int depth = 0; depth < theDepth; ++depth)
		{

			dimension = depth % numberOfDimensions;
			unsigned int firstIndexInDepth = (1 << depth) - 1;
			unsigned int maxDepth = (1 << depth);
			for (unsigned int indexInDepth = 0; indexInDepth < maxDepth;
					++indexInDepth)
			{
				unsigned int indexInArray = firstIndexInDepth + indexInDepth;
				unsigned int leftSonIndexInArray = 2 * indexInArray + 1;
				unsigned int rightSonIndexInArray = leftSonIndexInArray + 1;

				unsigned int whichElementInInterval = partition_complete_kdtree(
						theIntervalLength[indexInArray]);
				std::nth_element(
						thePoints.begin() + theIntervalMin[indexInArray],
						thePoints.begin() + theIntervalMin[indexInArray]
								+ whichElementInInterval,
						thePoints.begin() + theIntervalMin[indexInArray]
								+ theIntervalLength[indexInArray],
						[dimension](const FKDPoint<TYPE,numberOfDimensions> & a, const FKDPoint<TYPE,numberOfDimensions> & b) -> bool
						{
							if(a[dimension] == b[dimension])
							return a.getId() < b.getId();
							else
							return a[dimension] < b[dimension];
						});
				add_at_position(
						thePoints[theIntervalMin[indexInArray]
								+ whichElementInInterval], indexInArray);

				if (leftSonIndexInArray < theNumberOfPoints)
				{
					theIntervalMin[leftSonIndexInArray] =
							theIntervalMin[indexInArray];
					theIntervalLength[leftSonIndexInArray] =
							whichElementInInterval;
				}

				if (rightSonIndexInArray < theNumberOfPoints)
				{
					theIntervalMin[rightSonIndexInArray] =
							theIntervalMin[indexInArray]
									+ whichElementInInterval + 1;
					theIntervalLength[rightSonIndexInArray] =
							(theIntervalLength[indexInArray] - 1
									- whichElementInInterval);
				}
			}
		}

		dimension = theDepth % numberOfDimensions;
		unsigned int firstIndexInDepth = (1 << theDepth) - 1;
		for (unsigned int indexInArray = firstIndexInDepth;
				indexInArray < theNumberOfPoints; ++indexInArray)
		{
			add_at_position(thePoints[theIntervalMin[indexInArray]],
					indexInArray);

		}

	}

	std::size_t size() const
	{
		return theNumberOfPoints;
	}

private:

	unsigned int partition_complete_kdtree(unsigned int length)
	{
		if (length == 1)
			return 0;
		unsigned int index = 1 << (ilog2(length));


		if ((index / 2) - 1 <= length - index)
			return index - 1;
		else
			return length - index / 2;

	}

	unsigned int leftSonIndex(unsigned int index) const
	{
		return 2 * index + 1;
	}

	unsigned int rightSonIndex(unsigned int index) const
	{
		return 2 * index + 2;
	}

	bool intersects(unsigned int index,
			const FKDPoint<TYPE, numberOfDimensions>& minPoint,
			const FKDPoint<TYPE, numberOfDimensions>& maxPoint,
			int dimension) const
	{
		return (theDimensions[dimension][index] <= maxPoint[dimension]
				&& theDimensions[dimension][index] >= minPoint[dimension]);
	}

	bool is_in_the_box(unsigned int index,
			const FKDPoint<TYPE, numberOfDimensions>& minPoint,
			const FKDPoint<TYPE, numberOfDimensions>& maxPoint) const
	{
		for (unsigned int i = 0; i < numberOfDimensions; ++i)
		{
			if ((theDimensions[i][index] <= maxPoint[i]
					&& theDimensions[i][index] >= minPoint[i]) == false)
				return false;
		}

		return true;;
	}
	
	void add_at_position(const FKDPoint<TYPE, numberOfDimensions>& point,
			const unsigned int position)
	{
		for (unsigned int i = 0; i < numberOfDimensions; ++i)
			theDimensions[i][position] = point[i];
		theIds[position] = point.getId();

	}

	void add_at_position(FKDPoint<TYPE, numberOfDimensions> && point,
			const unsigned int position)
	{
		for (unsigned int i = 0; i < numberOfDimensions; ++i)
			theDimensions[i][position] = point[i];
		theIds[position] = point.getId();

	}

	unsigned int theNumberOfPoints;
	unsigned int theDepth;
	std::vector<FKDPoint<TYPE, numberOfDimensions> > thePoints;
	std::array<std::vector<TYPE>, numberOfDimensions> theDimensions;
	std::vector<unsigned int> theIntervalLength;
	std::vector<unsigned int> theIntervalMin;
	std::vector<unsigned int> theIds;
};

#endif /* FKDTREE_FKDTREE_H_ */
