#ifndef DetLayers_NavigationSchool_H
#define DetLayers_NavigationSchool_H

#include <vector>

#include "NavigableLayer.h"
#include "DetLayer.h"

/** A base class for NavigationSchools.
 *  The links between layers are computed or loaded from 
 *  persistent store by a NavigationSchool.
 *  The result is a container of NavigableLayers.
 */


class NavigationSchool {
public:

  NavigationSchool() : theAllDetLayersInSystem(0){}

  virtual ~NavigationSchool() {}

  typedef std::vector<NavigableLayer*>   StateType;

  virtual StateType navigableLayers() const = 0;


 /// Return the next (closest) layer(s) that can be reached in the specified
  /// NavigationDirection
  template<typename... Args>
  std::vector<const DetLayer*> 
  nextLayers(const DetLayer & detLayer, Args && ...args) const {
   assert( detLayer.seqNum()>=0);
   auto nl = theAllNavigableLayer[detLayer.seqNum()];
    return nl
      ? nl->nextLayers(std::forward<Args>(args)...)
      : std::vector<const DetLayer*>();
  }
  
  /// Returns all layers compatible 
  template<typename... Args>
  std::vector<const DetLayer*> 
  compatibleLayers(const DetLayer & detLayer, Args && ...args) const {
    auto nl = theAllNavigableLayer[detLayer.seqNum()];
    return nl
      ? nl->compatibleLayers(std::forward<Args>(args)...)
      : std::vector<const DetLayer*>();
  }

protected:

  void setState( const StateType& state) {
    for (auto nl : state)
      if (nl) theAllNavigableLayer[nl->detLayer()->seqNum()]=nl;
  }

  // index correspond to seqNum of DetLayers
  StateType theAllNavigableLayer;


  // will be obsoleted together with NAvigationSetter
public:
  const std::vector<DetLayer*> & allLayersInSystem() const {return *theAllDetLayersInSystem;}
 protected:
  const std::vector<DetLayer*> * theAllDetLayersInSystem;
};

#endif // NavigationSchool_H
