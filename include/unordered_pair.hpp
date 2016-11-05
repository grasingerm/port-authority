#ifndef __UNORDERED_PAIR_HPP__
#define __UNORDERED_PAIR_HPP__

template <class T> struct unordered_pair {
  T first;
  T second;
};

template <class T>
bool operator==(const unordered_pair<T> &up1, const unordered_pair<T> &up2) {
  if (up1.first == up2.first) {
    if (up1.second == up2.second)
      return true;
  } else if (up1.first == up2.second) {
    if (up1.second == up2.first)
      return true;
  }
  return false;
}

#endif
