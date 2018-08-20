#undef BEGIN_WRAP_COMMANDS
#undef END_WRAP_COMMANDS
#undef DEF_WRAP_COMMAND

#if defined(INSTANCE_COMMANDS)
#define BEGIN_WRAP_COMMANDS
#define END_WRAP_COMMANDS
#define DEF_WRAP_COMMAND(name)
#elif defined(BUILD_COMMANDS)
#define BEGIN_WRAP_COMMANDS						\
	struct PyMethodDef hip_methods[] =			\
	{

#define END_WRAP_COMMANDS						\
		{nullptr, nullptr, 0, nullptr}			\
	};

#define DEF_WRAP_COMMAND(name) {#name, PyWrap##name::execute, METH_VARARGS, PyWrap##name::docString},
#else
#define BEGIN_WRAP_COMMANDS
#define END_WRAP_COMMANDS
#define DEF_WRAP_COMMAND(name)									\
class PyWrap##name												\
{																\
public:															\
	static PyObject* execute(PyObject* self, PyObject* args);	\
	static const char docString[];								\
};
#endif
