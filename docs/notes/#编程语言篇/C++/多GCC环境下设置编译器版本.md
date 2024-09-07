这是centos7中安装gcc8以后，激活gcc8环境的脚本，source /opt/rh/devtoolset-8/enable。可以看到通过设置如下变量，使得编译时使用该指定版本gcc工具和相应的库文件。

```
export PATH=/opt/rh/devtoolset-8/root/usr/bin${PATH:+:${PATH}}
export MANPATH=/opt/rh/devtoolset-8/root/usr/share/man:${MANPATH}
export INFOPATH=/opt/rh/devtoolset-8/root/usr/share/info${INFOPATH:+:${INFOPATH}}
export PCP_DIR=/opt/rh/devtoolset-8/root
# Some perl Ext::MakeMaker versions install things under /usr/lib/perl5
# even though the system otherwise would go to /usr/lib64/perl5.
export PERL5LIB=/opt/rh/devtoolset-8/root//usr/lib64/perl5/vendor_perl:/opt/rh/devtoolset-8/root/usr/lib/perl5:/opt/rh/devtoolset-8/root//usr/share/perl5/vendor_perl${PERL5LIB:+:${PERL5LIB}}
# bz847911 workaround:
# we need to evaluate rpm's installed run-time % { _libdir }, not rpmbuild time
# or else /etc/ld.so.conf.d files?
rpmlibdir=$(rpm --eval "%{_libdir}")
# bz1017604: On 64-bit hosts, we should include also the 32-bit library path.
if [ "$rpmlibdir" != "${rpmlibdir/lib64/}" ]; then
  rpmlibdir32=":/opt/rh/devtoolset-8/root${rpmlibdir/lib64/lib}"
fi
export LD_LIBRARY_PATH=/opt/rh/devtoolset-8/root$rpmlibdir$rpmlibdir32${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=/opt/rh/devtoolset-8/root$rpmlibdir$rpmlibdir32:/opt/rh/devtoolset-8/root$rpmlibdir/dyninst$rpmlibdir32/dyninst${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# duplicate python site.py logic for sitepackages
pythonvers=2.7
export PYTHONPATH=/opt/rh/devtoolset-8/root/usr/lib64/python$pythonvers/site-packages:/opt/rh/devtoolset-8/root/usr/lib/python$pythonvers/site-packages${PYTHONPATH:+:${PYTHONPATH}}
export PKG_CONFIG_PATH=/opt/rh/devtoolset-8/root/usr/lib64/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}
```

```
GCC_VERSION="8.2"
export CC=/opt/compiler/gcc-${GCC_VERSION}/bin/gcc
export CXX=/opt/compiler/gcc-${GCC_VERSION}/bin/g++
export PATH=/opt/compiler/gcc-${GCC_VERSION}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/opt/compiler/gcc-${GCC_VERSION}/bin:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PKG_CONFIG_PATH=/opt/compiler/gcc-${GCC_VERSION}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}
```