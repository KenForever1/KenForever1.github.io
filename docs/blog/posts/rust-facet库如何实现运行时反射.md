---
title: rust-facet库如何实现运行时反射
date: 2025-08-02
authors: [KenForever1]
categories: 
  - Rust
labels: []
comments: true
---

<!-- more -->

## 从facet 反射实现

### 从例子看facet反射的两大功能

为序列化和反序列化，比如json、yaml服务。

#### Partial功能介绍

用于构建值，可以用于根据json、yaml等格式，反序列化构建rust类型以及设置成员的值。先通过Partial::alloc分配关于类型Outer的未初始化内存，然后通过.begin_field("name").set(String::from("Hello, world!"))为成员name设置值。

```rust
#[derive(Facet, PartialEq, Eq, Debug)]
struct Outer {
    name: String,
    inner: Inner,
}

#[derive(Facet, PartialEq, Eq, Debug)]
struct Inner {
    x: i32,
    b: i32,
}

#[test]
fn wip_struct_testleak1() {
    let v = Partial::alloc::<Outer>()
        .begin_field("name")
        .set(String::from("Hello, world!"))
        .end()
        .begin_field("inner")
        .begin_field("x")
        .set(42)
        .end()
        .begin_field("b")
        .set(43)
        .end()
        .end()
        .build();

    assert_eq!(
        *v,
        Outer {
            name: String::from("Hello, world!"),
            inner: Inner { x: 42, b: 43 }
        }
    );
}
```

#### PeekValue功能介绍

查看存在的值。例如：通过.field_by_name("number")获取number成员的值。

```rust
#[derive(Facet)]
struct TestStruct {
    number: i32,
    text: String,
}

#[test]
fn peek_struct() {
    // Create test struct instance
    let test_struct = TestStruct {
        number: 42,
        text: "hello".to_string(),
    };
    let peek_value = Peek::new(&test_struct);

    // Convert to struct and check we can convert to PeekStruct
    let peek_struct = peek_value
        .into_struct()
        .expect("Should be convertible to struct");

    // Test field access by name
    let number_field = peek_struct
        .field_by_name("number")
        .expect("Should have a number field");
    let text_field = peek_struct
        .field_by_name("text")
        .expect("Should have a text field");

    // Test field values
    let number_value = number_field.get::<i32>().unwrap();
    assert_eq!(*number_value, 42);

    let text_value = text_field.get::<String>().unwrap();
    assert_eq!(text_value, "hello");
}
```

### 核心存储结构

facet存储的核心数据结构，

+ SHAPE
+ VTABLE
```rust
pub unsafe trait Facet<'facet>: 'facet {
    /// The shape of this type
    ///
    /// Shape embeds all other constants of this trait.
    const SHAPE: &'static Shape;

    /// Function pointers to perform various operations: print the full type
    /// name (with generic type parameters), use the Display implementation,
    /// the Debug implementation, build a default value, clone, etc.
    ///
    /// If [`Self::SHAPE`] has `ShapeLayout::Unsized`, then the parent pointer needs to be passed.
    ///
    /// There are more specific vtables in variants of [`Def`]
    const VTABLE: &'static ValueVTable;
}
```

为core、alloc、std等模块中的数据结构都实现了Facet trait。以实现反射的两大功能。

### String类型如何实现trait
看个最简单的类型，从alloc模块中引入的String类型。对每个类型为了实现反射，都有一个字符串类型的描述标志，String类型就是"String"。
```rust
#[cfg(feature = "alloc")]
unsafe impl Facet<'_> for alloc::string::String {
    const VTABLE: &'static ValueVTable = &const {
        let mut vtable = value_vtable!(alloc::string::String, |f, _opts| write!(
            f,
            "{}",
            Self::SHAPE.type_identifier
        ));

        let vtable_sized = vtable.sized_mut().unwrap();
        // 如何解析String类型
        vtable_sized.parse = || {
            Some(|s, target| {
                // For String, parsing from a string is just copying the string
                Ok(unsafe { target.put(s.to_string()) })
            })
        };

        vtable
    };

    const SHAPE: &'static Shape = &const {
        Shape::builder_for_sized::<Self>()
            .def(Def::Scalar)
            .type_identifier("String")
            .ty(Type::User(UserType::Opaque))
            .build()
    };
}
```

### VTABLE如何实现

通过ValueVTableBuilder绑定了函数指针，通过Spez统一调用到对应的函数。
impl! crate
可以判断某个类型是否实现了某个 trait，[impls crate](https://docs.rs/impls/latest/impls/)。

```rust
#[macro_export]
macro_rules! value_vtable {
    ($type_name:ty, $type_name_fn:expr) => {
        const {
            $crate::ValueVTable::builder::<$type_name>()
                .type_name($type_name_fn)
                .display(|| {
                    if $crate::spez::impls!($type_name: core::fmt::Display) {
                        Some(|data, f| {
                            use $crate::spez::*;
                            (&&Spez(data)).spez_display(f)
                        })
                    } else {
                        None
                    }
                })
                // ...
                .build()
        }
    }
};        
```

spec模块实现了“自动解引用特化辅助工具”指的是通过自动解引用（auto-deref）技术，实现类似于specialization的功能。
简而言之，本模块让你可以根据类型实现的 trait 自动选择更合适的实现，且不需要用到 Rust 还未稳定的specialization feature功能。

[rust自动引用特化参考](https://github.com/dtolnay/case-studies/blob/master/autoref-specialization/README.md)。

例如，一个类型实现了 `Default` trait，那么 `spez_default_in_place` 方法会返回一个指向默认值的指针。将默认值写入到指定的内存位置，也就是PtrUninit。

```rust
//////////////////////////////////////////////////////////////////////////////////////
// Default (in place, because we can't have sized) 🏠🔄
//////////////////////////////////////////////////////////////////////////////////////

/// Specialization proxy for [`core::default::Default`]
pub trait SpezDefaultInPlaceYes {
    /// Creates a default value for the inner type in place.
    ///
    /// This method is called when the wrapped type implements `Default`.
    /// It writes the default value into the provided uninitialized memory.
    ///
    /// # Safety
    ///
    /// This function operates on uninitialized memory and requires that `target`
    /// has sufficient space allocated for type `T`.
    unsafe fn spez_default_in_place<'mem>(&self, target: PtrUninit<'mem>) -> PtrMut<'mem>;
}
impl<T: Default> SpezDefaultInPlaceYes for &SpezEmpty<T> {
    unsafe fn spez_default_in_place<'mem>(&self, target: PtrUninit<'mem>) -> PtrMut<'mem> {
        unsafe { target.put(<T as Default>::default()) }
    }
}

/// Specialization proxy for [`core::default::Default`]
pub trait SpezDefaultInPlaceNo {
    /// Fallback implementation when the type doesn't implement `Default`.
    ///
    /// This method is used as a fallback and is designed to be unreachable in practice.
    /// It's only selected when the wrapped type doesn't implement `Default`.
    ///
    /// # Safety
    ///
    /// This function is marked unsafe as it deals with uninitialized memory,
    /// but it should never be reachable in practice.
    unsafe fn spez_default_in_place<'mem>(&self, _target: PtrUninit<'mem>) -> PtrMut<'mem>;
}
impl<T> SpezDefaultInPlaceNo for SpezEmpty<T> {
    unsafe fn spez_default_in_place<'mem>(&self, _target: PtrUninit<'mem>) -> PtrMut<'mem> {
        unreachable!()
    }
}
```

ValueVTableBuilder是ValueVTable类型的构建器。
```rust
pub struct ValueVTableBuilder<T> {
    type_name: Option<TypeNameFn>,
    display: fn() -> Option<DisplayFn>,
    // ...
}
```
display函数实现就是通过transmute进行类型转换，从DisplayFnTyped类型转换为DisplayFn。

```rust
pub const fn display(mut self, display: fn() -> Option<DisplayFnTyped<T>>) -> Self {
    self.display = unsafe {
        mem::transmute::<fn() -> Option<DisplayFnTyped<T>>, fn() -> Option<DisplayFn>>(display)
    };
    self
}
```
```rust
/// Function to format a value for display
///
/// If both [`DisplayFn`] and [`ParseFn`] are set, we should be able to round-trip the value.
///
/// # Safety
///
/// The `value` parameter must point to aligned, initialized memory of the correct type.
pub type DisplayFn =
    for<'mem> unsafe fn(value: PtrConst<'mem>, f: &mut core::fmt::Formatter) -> core::fmt::Result;


/// Function to format a value for display
///
/// If both [`DisplayFn`] and [`ParseFn`] are set, we should be able to round-trip the value.
pub type DisplayFnTyped<T> = fn(value: &T, f: &mut core::fmt::Formatter) -> core::fmt::Result;****
```

### SHAPE如何实现

```rust
impl Shape {
    /// Returns a builder for a shape for some type `T`.
    pub const fn builder_for_sized<'a, T: Facet<'a>>() -> ShapeBuilder {
        ShapeBuilder::new(T::VTABLE)
            .layout(Layout::new::<T>())
            .id(ConstTypeId::of::<T>())
    }
    // ....
}
```

```rust
/// Builder for [`Shape`]
pub struct ShapeBuilder {
    id: Option<ConstTypeId>,
    layout: Option<ShapeLayout>, // 记录了layout信息
    vtable: &'static ValueVTable, // 记录了vtable信息，就是上面的value_vtable，存储了函数指针
    def: Def,
    ty: Option<Type>,
    type_identifier: Option<&'static str>,
    type_params: &'static [TypeParam],
    doc: &'static [&'static str],
    attributes: &'static [ShapeAttribute],
    type_tag: Option<&'static str>,
    inner: Option<fn() -> &'static Shape>,
}
```

layout是什么呢？实际上就是存储的core::alloc::Layout，记录了Size、Align等信息。可以分配一段该类型的未初始化内存，然后就可以反序列化时将内存写入到指定的内存中，就构造出了type的实例。
```rust
use core::alloc::Layout;

/// Schema for reflection of a type
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Shape {
    /// Unique type identifier, provided by the compiler.
    pub id: ConstTypeId,

    /// Size, alignment — enough to allocate a value of this type
    /// (but not initialize it.)
    pub layout: ShapeLayout,

    /// Function pointers to perform various operations: print the full type
    /// name (with generic type parameters), use the Display implementation,
    /// the Debug implementation, build a default value, clone, etc.
    ///
    /// If the shape has `ShapeLayout::Unsized`, then the parent pointer needs to be passed.
    ///
    /// There are more specific vtables in variants of [`Def`]
    pub vtable: &'static ValueVTable,
    // ...
}

/// Layout of the shape
#[derive(Clone, Copy, Debug, Hash)]
pub enum ShapeLayout {
    /// `Sized` type
    Sized(Layout),
    /// `!Sized` type
    Unsized,
}
```

#### 如何在堆上分配一段未初始化内存？

```rust
// facet/facet-core/src/types/mod.rs
impl Shape {
    /// Heap-allocate a value of this shape
    #[cfg(feature = "alloc")]
    #[inline]
    pub fn allocate(&self) -> Result<crate::ptr::PtrUninit<'static>, UnsizedError> {
        let layout = self.layout.sized_layout()?;

        Ok(crate::ptr::PtrUninit::new(if layout.size() == 0 {
            core::ptr::without_provenance_mut(layout.align())
        } else {
            // SAFETY: We have checked that layout's size is non-zero
            unsafe { alloc::alloc::alloc(layout) }
        }))
    }

    /// Deallocate a heap-allocated value of this shape
    #[cfg(feature = "alloc")]
    #[inline]
    pub unsafe fn deallocate_mut(&self, ptr: PtrMut) -> Result<(), UnsizedError> {
        use alloc::alloc::dealloc;

        let layout = self.layout.sized_layout()?;

        if layout.size() == 0 {
            // Nothing to deallocate
            return Ok(());
        }
        // SAFETY: The user guarantees ptr is valid and from allocate, we checked size isn't 0
        unsafe { dealloc(ptr.as_mut_byte_ptr(), layout) }

        Ok(())
    }
    // ...
}
```

什么时候会用到根据layout分配内存呢？在facet reflect的Partial功能中。还记得最前面的Partial就是调用了alloc函数分配Outer类型的内存吗？
```rust
impl<'facet> Partial<'facet> {
    /// Allocates a new Partial instance with the given shape
    pub fn alloc_shape(shape: &'static Shape) -> Result<Self, ReflectError> {
        crate::trace!(
            "alloc_shape({:?}), with layout {:?}",
            shape,
            shape.layout.sized_layout()
        );

        let data = shape.allocate().map_err(|_| ReflectError::Unsized {
            shape,
            operation: "alloc_shape",
        })?;
        // ....
    }

    /// Allocates a new TypedPartial instance with the given shape and type
    pub fn alloc<T>() -> Result<TypedPartial<'facet, T>, ReflectError>
    where
        T: Facet<'facet>,
    {
        Ok(TypedPartial {
            inner: Self::alloc_shape(T::SHAPE)?,
            phantom: PhantomData,
        })
    }
}
```

### Partial的实现原理

Partial通过一个Vec\<Frame\>来保存操作的元素。以Partial构造一个Struct为例。当前操作name field，那么Vec保存的最后一帧就是记录的name fileld的信息，包括shape、数据指针、状态Tracker等。
```rust
pub struct Partial<'facet> {
    /// stack of frames to keep track of deeply nested initialization
    frames: Vec<Frame>,
    // ...
}

struct Frame {
    /// Address of the value being initialized
    data: PtrUninit<'static>,

    /// Shape of the value being initialized
    shape: &'static Shape,

    /// Tracks initialized fields
    tracker: Tracker,

    /// Whether this frame owns the allocation or is just a field pointer
    ownership: FrameOwnership,
}
```
从前面的例子看
```rust
#[derive(Facet, PartialEq, Eq, Debug)]
struct Outer {
    name: String,
}

#[test]
fn wip_struct_testleak1() {
    let v = Partial::alloc::<Outer>()
        .begin_field("name")
        .set(String::from("Hello, world!"))
        .end()
    // ......
}
```

首先,alloc分配Outer类型我们已经介绍清楚了。从begin_field说起。
根据field_name, 例如Outer 结构体的name字段，先获取Struct的struct_type.fields，然后获取到一个index。就可以根据index去获取这个field的信息了。我们的目的是获取field的Shape和未初始化指针，后续就可以通过set函数设置该字段的值。

self.frames.last_mut()就是获取当前栈帧。当前操作的是Struct类型的栈帧。
```rust
// /home/ken/tmp/facet/facet-reflect/src/partial/mod.rs
/// Selects a field of a struct with a given name
pub fn begin_field(&mut self, field_name: &str) -> Result<&mut Self, ReflectError> {
    let frame = self.frames.last_mut().unwrap();
    match frame.shape.ty {
        Type::User(user_type) => match user_type {
            UserType::Struct(struct_type) => {
                let idx = struct_type.fields.iter().position(|f| f.name == field_name);
                let idx = match idx {
                    Some(idx) => idx,
                    None => {
                        return Err(ReflectError::OperationFailed {
                            shape: frame.shape,
                            operation: "field not found",
                        });
                    }
                };
                self.begin_nth_field(idx)
            }
        },
    }
}
```

调用begin_nth_field函数时，最后一个Frame还是Struct的信息。
在找到name feild后，如果已经初始化，就先drop初始化的数据，然后后续重新初始化。同时，更新最后一帧，也就是last_frame为当前操作的Struct的name field。
```rust
 /// Selects the nth field of a struct by index
    pub fn begin_nth_field(&mut self, idx: usize) -> Result<&mut Self, ReflectError> {
        let frame = self.frames.last_mut().unwrap();
        match frame.shape.ty {
            Type::User(user_type) => match user_type {
                UserType::Struct(struct_type) => {
                    if idx >= struct_type.fields.len() {
                        return Err(ReflectError::OperationFailed {
                            shape: frame.shape,
                            operation: "field index out of bounds",
                        });
                    }
                    let field = &struct_type.fields[idx];

                    match &mut frame.tracker {
                        Tracker::Uninit => {
                            frame.tracker = Tracker::Struct {
                                iset: ISet::new(struct_type.fields.len()),
                                current_child: Some(idx),
                            }
                        },
                        Tracker::Struct {
                            iset,
                            current_child,
                        } => {
                            // Check if this field was already initialized
                            if iset.get(idx) {
                                // Drop the existing value before re-initializing
                                // 获取name字段的指针，drop已经初始化的数据，进行重新初始化
                                let field_ptr = unsafe { frame.data.field_init_at(field.offset) };
                                if let Some(drop_fn) =
                                    field.shape.vtable.sized().and_then(|v| (v.drop_in_place)())
                                {
                                    unsafe { drop_fn(field_ptr) };
                                }
                                // Unset the bit so we can re-initialize
                                iset.unset(idx);
                            }
                            *current_child = Some(idx);
                        }
                        _ => unreachable!(),
                    }

                    // Push a new frame for this field onto the frames stack.
                    let field_ptr = unsafe { frame.data.field_uninit_at(field.offset) };
                    let field_shape = field.shape;
                    // 修改最后一帧，也就是last_frame为当前操作的Struct的name field
                    self.frames
                        .push(Frame::new(field_ptr, field_shape, FrameOwnership::Field));

                    Ok(self)
                }
            },
        }
    }
```

set设置值
```rust

/// Sets a value wholesale into the current frame
pub fn set<U>(&mut self, value: U) -> Result<&mut Self, ReflectError>
where
    U: Facet<'facet>,
{
    self.require_active()?;

    // For conversion frames, store the value in the conversion frame itself
    // The conversion will happen during end()
    let ptr_const = PtrConst::new(&raw const value);
    unsafe {
        // Safety: We are calling set_shape with a valid shape and a valid pointer
        self.set_shape(ptr_const, U::SHAPE)?
    };

    // Prevent the value from being dropped since we've copied it
    core::mem::forget(value);
    Ok(self)
}
```
set函数调用了set_shape函数，因为begin_field函数接着调用的就是set函数，因此set_shape函数中获取的last_mut frame就是name字段的信息。然后就为data指针指向的数据赋值了。
```rust
/// Sets a value into the current frame by shape, for shape-based operations
#[inline]
pub unsafe fn set_shape(
    &mut self,
    src_value: PtrConst<'_>,
    src_shape: &'static Shape,
) -> Result<&mut Self, ReflectError> {
    let fr = self.frames.last_mut().unwrap();

    unsafe {
        fr.data.copy_from(src_value, fr.shape).unwrap();
    }
    fr.tracker = Tracker::Init;
    Ok(self)
}
```
到此，成功实现了Struct的Partial功能，从分配未初始化的内存，然后通过string类型的field名称，构造一个个field，最终构造出一个结构体。

