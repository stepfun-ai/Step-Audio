eval $(curl -s deploy.i.shaipower.com/httpproxy)

docker build -f Dockerfile-vllm -t test-audio:12.4.1 . \
        --build-arg http_proxy="$http_proxy" \
        --build-arg https_proxy="$https_proxy" \
        --build-arg no_proxy="$no_proxy"
