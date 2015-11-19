
#ifdef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
enum cudaMemcpyKind {
	cudaMemcpyDeviceToDevice,
	cudaMemcpyDeviceToHost,
	cudaMemcpyHostToDevice
};
#endif
