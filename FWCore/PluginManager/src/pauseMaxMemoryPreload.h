#ifndef FWCore_PluginManager_src_pauseMaxMemoryPreload_h
#define FWCore_PluginManager_src_pauseMaxMemoryPreload_h

// By default do nothing, but add "hooks" that MaxMemoryPreload can
// override with LD_PRELOAD
void pauseMaxMemoryPreload();
void unpauseMaxMemoryPreload();

#endif
