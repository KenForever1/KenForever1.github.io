### kotlin usage

#### gson

+ 自定义类的序列化和反序列化
```
class MenuContentInterfaceAdapter : JsonDeserializer<Any>, JsonSerializer<Any> {

    companion object {
        const val CLASSNAME = "CLASSNAME"
        const val DATA  = "DATA"
    }

    @Throws(JsonParseException::class)
    override fun deserialize(jsonElement: JsonElement, type: Type,
                             jsonDeserializationContext: JsonDeserializationContext): Any {

        val jsonObject = jsonElement.asJsonObject
        val prim = jsonObject.get(CLASSNAME) as JsonPrimitive
        val className = prim.asString
        val objectClass = getObjectClass(className)
        return jsonDeserializationContext.deserialize(jsonObject.get(DATA), objectClass)
    }

    override fun serialize(jsonElement: Any, type: Type, jsonSerializationContext: JsonSerializationContext): JsonElement {
        val jsonObject = JsonObject()
        jsonObject.addProperty(CLASSNAME, jsonElement.javaClass.name)
        jsonObject.add(DATA, jsonSerializationContext.serialize(jsonElement))
        return jsonObject
    }

    private fun getObjectClass(className: String): Class<*> {
        try {
            return Class.forName(className)
        } catch (e: ClassNotFoundException) {
            throw JsonParseException(e.message)
        }

    }
}
```

```
private var gson : Gson? = null
val gsonBuilder = GsonBuilder()
gsonBuilder.registerTypeAdapter(MenuContent::class.java, MenuContentInterfaceAdapter())
gson = gsonBuilder.create()
```
https://paul-stanescu.medium.com/custom-interface-adapter-to-serialize-and-deserialize-interfaces-in-kotlin-using-gson-8539c04b4c8f

+ enum 序列化成数字
// https://ejin66.github.io/2018/12/19/gson-enum.html
```
import com.google.gson.Gson
import com.google.gson.TypeAdapter
import com.google.gson.TypeAdapterFactory
import com.google.gson.annotations.SerializedName
import com.google.gson.reflect.TypeToken
import com.google.gson.stream.JsonReader
import com.google.gson.stream.JsonToken
import com.google.gson.stream.JsonWriter

// https://ejin66.github.io/2018/12/19/gson-enum.html

class EnumTypeAdapterFactory: TypeAdapterFactory {
    override fun <T : Any> create(gson: Gson, type: TypeToken<T>): TypeAdapter<T>? {
        if (!type.rawType.isEnum) {
            return null
        }

        val maps = mutableMapOf<T, ValueType>()

        type.rawType.enumConstants.filter { it != null }.forEach {
            val tt: T = it!! as T

            val serializedName = tt.javaClass.getField(it.toString()).getAnnotation(SerializedName::class.java)

            if (serializedName != null) {
                maps[tt] = ValueType(serializedName.value, BasicType.STRING)
                return@forEach
            }

            val field = tt.javaClass.declaredFields.firstOrNull { it2 ->
                BasicType.isBasicType(it2.type.name)
            }
            if (field != null) {
                field.isAccessible = true
                val basicType = BasicType.get(field.type.name)
                val value: Any = when (basicType) {
                    BasicType.INT -> field.getInt(tt)
                    BasicType.STRING -> field.get(tt) as String
                    BasicType.LONG -> field.getLong(tt)
                    BasicType.DOUBLE -> field.getDouble(tt)
                    BasicType.BOOLEAN -> field.getBoolean(tt)
                }
                maps[tt] = ValueType(value, basicType)
            } else {
                maps[tt] = ValueType(tt.toString(), BasicType.STRING)
            }
        }

        return object: TypeAdapter<T>() {
            override fun write(out: JsonWriter, value: T?) {
                if (value == null) {
                    out.nullValue()
                } else {
                    val valueType = maps[value]!!
                    when (valueType.type) {
                        BasicType.INT -> out.value(valueType.value as Int)
                        BasicType.STRING -> out.value(valueType.value as String)
                        BasicType.LONG -> out.value(valueType.value as Long)
                        BasicType.DOUBLE -> out.value(valueType.value as Double)
                        BasicType.BOOLEAN -> out.value(valueType.value as Boolean)
                    }
                }
            }

            override fun read(reader: JsonReader): T? {
                if (reader.peek() == JsonToken.NULL) {
                    reader.nextNull()
                    return null
                } else {
                    val source = reader.nextString()
                    var tt: T? = null
                    maps.forEach { value, type ->
                        if (type.value.toString() == source) {
                            tt = value
                            return@forEach
                        }
                    }
                    return tt
                }
            }

        }
    }

    data class ValueType(var value: Any, var type: BasicType)

    enum class BasicType(var value: String) {
        INT("int"),
        STRING("java.lang.String"),
        LONG("long"),
        DOUBLE("double"),
        BOOLEAN("boolean");


        companion object {
            fun isBasicType(name: String): Boolean {
                return values().any { it.value == name }
            }

            fun get(name: String) = values().first { it.value == name }
        }
    }
}
```

```
gson = new GsonBuilder().registerTypeAdapterFactory(new EnumTypeAdapterFactory()).create();
```

#### tcp client

```
import java.io.*;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class TCPClient {
    String ip;
    int port;

    DataInputStream inFromServer;
    DataOutputStream outToServer;

    Socket clientSocket;

    public Boolean connect(String _ip, int _port) {
        ip = _ip;
        port = _port;
        try {
            //create client socket, connect to server
//            clientSocket = new Socket();
//            SocketAddress endpoint = new InetSocketAddress(this.ip, this.port);
//            clientSocket.connect(endpoint);
            clientSocket = new Socket(this.ip, this.port);


            //create output stream attached to socket
            outToServer = new DataOutputStream(clientSocket.getOutputStream());

            //create input stream attached to socket
            inFromServer = new DataInputStream(clientSocket.getInputStream());

        } catch (Exception ex) {

        }
        return true;
    }

    public byte[] recv(int len) throws IOException {
        ByteArrayOutputStream res = new ByteArrayOutputStream();
        ByteBuffer byteBuffer = ByteBuffer.allocate(len);

        int dataLen = len;
        while (dataLen != 0) {

            int r = inFromServer.read(byteBuffer.array());
            dataLen -= r;

            byte[] b = new byte[r];
            byteBuffer.get(b, 0, r);
            res.write(b);
        }
        return res.toByteArray();
    }


    public Boolean send(byte[] data) throws IOException {

        int dataLen = data.length;
        ByteBuffer b = ByteBuffer.allocate(4);
        b.order(ByteOrder.BIG_ENDIAN);
        b.putInt(dataLen);

//        ByteArrayOutputStream output = new ByteArrayOutputStream();
//        output.write(b.array());
//        output.write(data);
//
//        //send line to server
//        outToServer.write(output.toByteArray());

        byte[] bytesAll = new byte[4 + data.length];
        ByteBuffer buffer = ByteBuffer.wrap(bytesAll);
        buffer.put(b.array());
        buffer.put(data);

        //send line to server
        outToServer.write(buffer.array());
        outToServer.flush();
        return true;
    }
}
```


