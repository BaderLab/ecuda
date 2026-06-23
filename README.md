<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache 2.0][license-shield]][license-url]
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->

<br/>
<div align="center">
  <a href="https://github.com/BaderLab/ecuda">
    <img src="./docs/ecuda-logo.svg" width="110" alt="ecuda logo">
  </a>

  <h3 align="center">ecuda</h3>
  <!-- <h1 align="center">ecuda</h1> -->

  <p align="center">
    STL-style abstractions for CUDA.
	<br/>
	<a href="https://github.com/BaderLab/ecuda"><strong>Explore the docs »</strong></a>
	<br/>
	<br/>
	<a href="https://github.com/BaderLab/ecuda">View Demo</a>
	&middot;
	<a href="https://github.com/BaderLab/ecuda/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
	&middot;
	<a href="https://github.com/BaderLab/ecuda/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>

</div>

<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
	  <a href="#about-the-project">About The Project</a>
	</li>
	<li>
	  <a href="#getting-started">Getting Started</a>
	  <ul>
	    <li><a href="#prerequistes">Prerequisites</a></li>
		<li><a href="#installation">Installation</a></li>
	  </ul>
	</li>
	<li>
	  <a href="#usage">Usage</a>
	</li>
	<li><a href="#roadmap">Roadmap</a></li>
	<li><a href="#contributing">Contributing</a></li>
	<li><a href="#license">License</a></li>
	<li><a href="#contact">Contant</a></li>
	<li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

**ecuda** is a C++ wrapper around the CUDA C API designed to closely resemble and
be functionally equivalent to the C++ Standard Template Library (STL).
Specifically: algorithms, containers, and iterators. These elements play nice
with host containers and can be used in device code.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![C++][C++]][C++-url]
* [![CUDA][CUDA]][CUDA-url]
* [![doctest][doctest]][doctest-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* CUDA
  ```sh
  sudo apt-get -y install cuda # Debian
  sudo pacman -S cuda # Arch
  ```

### Installation

#### Option 1

1. Clone the repo
```sh
git clone https://github.com/BaderLab/ecuda
```
2. Compile and run the tests (optional)
```sh
cd ecuda
mkdir build
cmake -DECUDA_BUILD_TESTS=ON
make
ctest --output-on-failure
```

#### Option 2

Use FetchContent to add to your own project's `CMakeLists.txt`.

```cmake
# the start of your CMakeLists.txt...

include( FetchContent )

FetchContent_Declare(
    ecuda
    GIT_REPOSITORY https://github.com/BaderLab/ecuda
    GIT_TAG        "master"
    SOURCE_DIR     "${CMAKE_BINARY_DIR}/_deps/ecuda-src"
    BINARY_DIR     "${CMAKE_BINARY_DIR}/_deps/ecuda-build"
)

FetchContent_MakeAvailable( ecuda )

find_package( CUDAToolkit REQUIRED )

# ... the rest of your CMakeLists.txt

target_link_libraries( YourExecutable PUBLIC ecuda::ecuda PRIVATE CUDA::cudart )
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

```cpp
#include <ecuda.hpp>
```

```cpp
template<class Container>
__global__
void
reverse_order(
  typename Container::const_kernel_argument in,
  typename Container::kernel_argument out
)
{
  const int t = threadIdx.x;
  if( t < in.size() ) {
    auto value = *(in.begin()+t);
	*(out.begin()+(out.size()-t-1)) = value;
  }
}
```

```cpp
std::vector<double> hostVector( 1000 );
// ... fill hostVector with data
ecuda::vector<double> deviceVector1( hostVector.begin(), hostVector.end() );
ecuda::vector<double> deviceVector2( 1000 );
CUDA_CALL_KERNEL_AND_WAIT( reverse_order<<<1,1000>>>( deviceVector1, deviceVector2 ) );
ecuda::copy( deviceVector2.begin(), deviceVector2.end(), hostVector.begin() );
```

```cpp
std::vector<double> hostMatrix( 10*10 );
// ... fill hostMatrix with data
ecuda::matrix<double> deviceMatrix1( 10, 10 );
ecuda::matrix<double> deviceMatrix2( 10, 10 );
ecuda::copy( hostMatrix.begin(), hostMatrix.end(), deviceMatrix1.begin() );
CUDA_CALL_KERNEL_AND_WAIT( reverse_order<<<1,10*10>>>( deviceMatrix1, deviceMatrix2 ) );
ecuda::copy( deviceMatrix2.begin(), deviceMatrix2.end(), hostVector.begin() );
```

```cpp
std::vector<double> hostCube( 10*10*10 );
// ... fill hostCube with data
ecuda::cube<double> deviceCube1( 10, 10, 10 );
ecuda::cube<double> deviceCube2( 10, 10, 10 );
ecuda::copy( hostCube.begin(), hostCube.end(), deviceCube1.begin() );
CUDA_CALL_KERNEL_AND_WAIT( reverse_order<<<1,10*10>>>( deviceCube1, deviceCube2 ) );
ecuda::copy( deviceCube2.begin(), deviceCube2.end(), hostVector.begin() );
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

No additional features are planned.

See the [open issues](https://github.com/BaderLab/ecuda/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/BaderLab/ecuda/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=BaderLab/ecuda" alt="contrib.rocks image"/>
</a>

<!-- LICENSE -->
## License

Distributed under the Apache 2.0 license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Scott D. Zuyderduyn - scott.zuyderduyn@utoronto.ca

Project Link: [https://github.com/BaderLab/ecuda](https://github.com/BaderLab/ecuda)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* README based on the [othneildrew/Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/BaderLab/ecuda.svg?style=for-the-badge
[contributors-url]: https://github.com/BaderLab/ecuda/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/BaderLab/ecuda.svg?style=for-the-badge
[forks-url]: https://github.com/BaderLab/ecuda/network/members
[stars-shield]: https://img.shields.io/github/stars/BaderLab/ecuda.svg?style=for-the-badge
[stars-url]: https://github.com/BaderLab/ecuda/stargazers
[issues-shield]: https://img.shields.io/github/issues/BaderLab/ecuda.svg?style=for-the-badge
[issues-url]: https://github.com/BaderLab/ecuda/issues
[license-shield]: https://img.shields.io/github/license/BaderLab/ecuda.svg?style=for-the-badge
[license-url]: https://github.com/BaderLab/ecuda/blob/master/LICENSE.txt
[product-screenshot]: images/screenshot.png
<!-- Shields.io badges. You can a comprehensive list with many more badges at: https://github.com/inttter/md-badges -->
[C++]: https://img.shields.io/badge/C++-%2300599C.svg?logo=c%2B%2B&logoColor=white
[C++-url]: https://isocpp.org/
[CUDA]: https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=fff
[CUDA-url]: https://developer.nvidia.com/cuda-downloads
[doctest]: https://img.shields.io/badge/doctest-na.svg?labelColor=grey&color=grey&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbDpzcGFjZT0icHJlc2VydmUiIHdpZHRoPSIyNC4zMDk3bW0iIGhlaWdodD0iMjcuNDI0NG1tIiB2ZXJzaW9uPSIxLjEiIHN0eWxlPSJzaGFwZS1yZW5kZXJpbmc6Z2VvbWV0cmljUHJlY2lzaW9uOyB0ZXh0LXJlbmRlcmluZzpnZW9tZXRyaWNQcmVjaXNpb247IGltYWdlLXJlbmRlcmluZzpvcHRpbWl6ZVF1YWxpdHk7IGZpbGwtcnVsZTpldmVub2RkOyBjbGlwLXJ1bGU6ZXZlbm9kZCINCnZpZXdCb3g9IjAgMCAyMjAyIDI0ODQiDQogeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPg0KIDxkZWZzPg0KICA8c3R5bGUgdHlwZT0idGV4dC9jc3MiPg0KICAgPCFbQ0RBVEFbDQogICAgLnN0cjAge3N0cm9rZTojMzkzNTZDO3N0cm9rZS13aWR0aDo5My42ODg0O3N0cm9rZS1saW5lY2FwOnJvdW5kO3N0cm9rZS1saW5lam9pbjpyb3VuZH0NCiAgICAuZmlsMCB7ZmlsbDpub25lfQ0KICAgIC5maWwxIHtmaWxsOiNGRUZFRkV9DQogICAgLmZpbDIge2ZpbGw6IzM5MzU2Q30NCiAgICAuZmlsMyB7ZmlsbDojNUVFMDQ1fQ0KICAgXV0+DQogIDwvc3R5bGU+DQogPC9kZWZzPg0KIDxnIGlkPSJDYXBhX3gwMDIwXzEiPg0KICA8bWV0YWRhdGEgaWQ9IkNvcmVsQ29ycElEXzBDb3JlbC1MYXllciIvPg0KICA8cGF0aCBjbGFzcz0iZmlsMCBzdHIwIiBkPSJNMjE5IDE1OGwxNzY0IDBjOTUsMCAxNzIsNzcgMTcyLDE3MWwwIDE3NjVjMCw5NCAtNzcsMTcyIC0xNzIsMTcybC0xNzY0IDBjLTk1LDAgLTE3MiwtNzggLTE3MiwtMTcybDAgLTE3NjVjMCwtOTQgNzcsLTE3MSAxNzIsLTE3MXoiLz4NCiAgPGcgaWQ9Il8yODMxNDA4NjM5ODA4Ij4NCiAgIDxwYXRoIGNsYXNzPSJmaWwxIiBkPSJNMTU1NiA0MjhjLTQ4LC0yNyAtOTcsLTUyIC0xNDgsLTczIC0yMDcsLTg1IC00MjksLTkzIC01NjIsMTE2IC0yMDIsMzE3IC0zMjIsNjU1IC0zNDQsMTAzMSAtMTgsMzExIDMzLDYxNCAxMDEsOTE2IDIyLDEwMCAxNzEsODEgMTY3LC0yMiAtNCwtODEgLTE5LC0xNjIgLTMxLC0yNDMgLTE5LC0xMjIgLTQxLC0yNzcgNTQsLTM3NSAyNjEsLTI2OSA0NTIsLTQzMCA0NDQsLTgzNiAtMSwtNzEgMTMsLTE0MCA2MSwtMTk0IDgwLC04OCAyMzMsLTExMiAzNDUsLTExOCAxMDYsLTYgMjEzLDIgMzE5LDEzbDEgMWMzMiwzIDYyLC0xMSA3OSwtMzggMzEsLTUwIDcsLTExNiAtNTEsLTEyOSAtMTQzLC0zMyAtMjg4LC01MyAtNDM1LC00OXoiLz4NCiAgIDxwYXRoIGNsYXNzPSJmaWwyIiBkPSJNMTU0MyA0ODJjLTIwMywtMTE5IC00OTUsLTIyOCAtNjUyLDE4IC0xOTcsMzA5IC0zMTQsNjM4IC0zMzUsMTAwNSAtMTgsMzA3IDMyLDYwNCA5OSw5MDIgOCwzNyA2MywzMCA2MiwtOCAtNCwtODAgLTE5LC0xNTkgLTMxLC0yMzcgLTIxLC0xNDMgLTQxLC0zMDggNjgsLTQyMSAyNTAsLTI1NyA0MzgsLTQxMCA0MzAsLTc5OCAtMiwtODUgMTcsLTE2NyA3NCwtMjMxIDE1MSwtMTY3IDUwOCwtMTQ0IDcxMCwtMTIxbDAgMGMzOSwzIDQ4LC01NCAxMCwtNjIgLTE0MywtMzQgLTI4OCwtNTMgLTQzNSwtNDd6Ii8+DQogICA8cGF0aCBjbGFzcz0iZmlsMiIgZD0iTTczMiAxNzE5Yy0xNzMsMTc4IC01Niw0NTcgLTQ2LDY4MSAwLDEgLTEsMCAtMSwwIC0xMjcsLTU2OCAtMjEzLC0xMTg1IDIzMiwtMTg4MyAxMzQsLTIxMSAzODYsLTE0MSA2MTgsLTMgMTQ4LC04IDI5MywxMSA0MzYsNDYgMCwwIDAsMCAwLDAgLTQyOSwtNDkgLTgyNiwtMiAtODE4LDM4MyA3LDM3OCAtMTc4LDUyNiAtNDIxLDc3NnoiLz4NCiAgIDxwYXRoIGNsYXNzPSJmaWwxIiBkPSJNNjg0IDE1NzljLTEwNSw5MSAtMjY0LDEzNCAtMzkyLDE4NCAtMzAsMTEgLTUxLDM4IC01NCw2OSAtNiw1NyA0NCwxMDQgMTAxLDkybDIgMGMzMDMsLTcyIDY5MSwtMjAyIDg0NCwtNDk3IDYwLC0xMTYgNzcsLTI0MiA2MywtMzcxIC0zMiwtMjkxIC0yMDksLTQ4NSAtNDc1LC01OTUgLTE5OCwtODEgLTM1NSwtMTk3IC00MzIsLTQwNSAtMzAsLTc5IC0xNDQsLTczIC0xNjUsOSAtMTExLDQzNCA2NCw3OTcgNDA0LDEwNjcgNzEsNTcgMjAyLDE2MSAyMTQsMjU2IDEwLDc1IC01OCwxNDYgLTExMCwxOTF6Ii8+DQogICA8cGF0aCBjbGFzcz0iZmlsMyIgZD0iTTcxOSAxNjE5Yy0xMTIsOTcgLTI3MiwxNDEgLTQwOCwxOTQgLTM2LDEzIC0yMCw2NyAxNyw1OWwxIDBjMjg0LC02NyA2NjQsLTE5MSA4MDksLTQ3MCA1NSwtMTA2IDY5LC0yMjIgNTcsLTM0MCAtMzAsLTI3MSAtMTkzLC00NDkgLTQ0MiwtNTUxIC0yMTMsLTg4IC0zODAsLTIxNSAtNDYyLC00MzYgLTEyLC0zMSAtNTUsLTI5IC02MywzIC0xMDUsNDEyIDYwLDc1NSAzODYsMTAxMyA4NCw2NyAyMTgsMTc2IDIzMywyOTAgMTMsOTcgLTYwLDE3OSAtMTI4LDIzOHoiLz4NCiAgIDxwYXRoIGNsYXNzPSJmaWwzIiBkPSJNNzQwIDE2NDJjLTExNiwxMDAgLTI3MiwxNDMgLTQxOCwyMDAgMCwwIDAsMCAwLDAgNTQ2LC0xMzAgODg4LC0zNDkgODQyLC03NzcgLTI0LC0yMTUgLTEzOSwtNDA5IC00MjMsLTUyNiAtMjM1LC05NiAtMzk4LC0yMzQgLTQ3OSwtNDUzIC0xLC0yIC0zLC0yIC00LDAgLTkyLDM2MSAyMiw3MDEgMzc1LDk4MCAyMTEsMTY4IDM3NiwzNDQgMTA3LDU3NnoiLz4NCiAgPC9nPg0KICA8cGF0aCBjbGFzcz0iZmlsMyIgZD0iTTEwNTkgMTczNWM4NywtMTMgMTM4LC02MSAxNDYsLTE0NiA5LC05MCAxNTIsLTg3IDE1MywwIDEsMTA5IDYzLDE0MyAxNDcsMTQ2IDk3LDMgOTEsMTQ3IDAsMTUzIC05NCw3IC0xMzgsNjEgLTE0NywxNDcgLTEwLDk1IC0xNTMsOTQgLTE1MywwIDAsLTkxIC00OCwtMTUxIC0xNDYsLTE0NyAtODAsNCAtODcsLTE0MCAwLC0xNTN6Ii8+DQogIDxwYXRoIGNsYXNzPSJmaWwyIiBkPSJNMTUwMSAxNDY0Yzg4LC0xMyAxMzgsLTYxIDE0NiwtMTQ3IDksLTkwIDE1MiwtODYgMTUzLDAgMSwxMTAgNjQsMTQ0IDE0NywxNDcgOTcsMyA5MSwxNDYgMCwxNTMgLTk0LDcgLTEzOCw2MCAtMTQ3LDE0NiAtMTAsOTUgLTE1Myw5NCAtMTUzLDAgMCwtOTAgLTQ4LC0xNTAgLTE0NiwtMTQ2IC04MCwzIC04NiwtMTQwIDAsLTE1M3oiLz4NCiA8L2c+DQo8L3N2Zz4NCg==
[doctest-url]: https://github.com/doctest/doctest
